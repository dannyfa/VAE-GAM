"""
Z-based fMRIVAE regression model w/ task as a real variable (i.e, boxcar * HRF)
- Added single voxel noise modeling (epsilon param)
- Added motion regressors in 6 degrees of freedom (from fmriprep)
- Added 1D GPs to model regressors (task + 6 motion params)
- Added initilization using and avg of SPM's task beta map slcd to take only 11% of total explained variance
- Added L1 regularization to all covariate maps. This helps correcting spurious signals.
- Fixed GP plotting issues
- Testing a version w/ out ANY HRF convolution
  - Simply feeds bin task covariate to GP to get yq
  - Then adds these to same bin coveriate
  - Use result of above to scale effect map.

To Do's
- Consider other 'cheaper' init options.
- Mk model able to handle init and noinit versions.
- Add sMC for time series modeling.
- Add time dependent latent space plotting. And improve overall visual quality of LS plot ...
- Mk it flexible enough to automate transference to other dsets.
"""

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import nibabel as nib
import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal, Normal, kl
#uncomment if needing to chase-down a nan in loss
#from torch import autograd
from umap import UMAP
import os
import itertools
from sklearn.decomposition import PCA
import gp
import pandas as pd
from scipy.stats import gamma # for HRF funct
import copy #for deep copy of xq array

#define HRF here ...
#should  be able to import it though from preproc module
#HRF funct.
def hrf(times):
    """ Return values for HRF at given times """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6


#am also defining linW KL computing function here
#this will also be imported through a separate module
def calc_linW_KL(sa, va):
    """
    Computes KL for linear weight term (kappa_\alpha)
    This term is added to KL stemming from GP itself, to yield a total GP
    contribution to the VAE-GAM objective.
    Args:
    sa --> posterior mean
    va --> posterior std
    Outputs:
    KL between prior (N(0,1)) and posterior (N(sa, va^2))
    This is computed as in the first term of Eqn 9 of Appendix B.
    """
    kl = 0.5 * ((1/torch.pow(va,2)) + (torch.pow(sa, 2)/torch.pow(va,2)) + \
    torch.log(torch.pow(va,2)))
    return kl

# maintained shape of original nn by downsampling data on preprocessing
IMG_SHAPE = (41,49,35)
IMG_DIM = np.prod(IMG_SHAPE)

class VAE(nn.Module):
    def __init__(self, nf=8, save_dir='', lr=1e-3, num_covariates=7, num_latents=32, device_name="auto", task_init = '', num_inducing_pts=6, mll_scale=10.0, l1_scale=1.0):
        super(VAE, self).__init__()
        self.nf = nf
        self.save_dir = save_dir
        self.lr = lr
        self.num_covariates = num_covariates
        self.num_latents = num_latents
        self.z_dim = self.num_latents + self.num_covariates + 1
        #adding l1_scale for regularization term
        self.l1_scale = l1_scale
        assert device_name != "cuda" or torch.cuda.is_available()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device_name)
        if self.save_dir != '' and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
		# init epsilon param modeling single voxel variance
		# -log(10) initial value accounts for removing model_precision term from original version
        epsilon = -np.log(10)*torch.ones([IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]], dtype=torch.float64, device = self.device)
        self.epsilon = torch.nn.Parameter(epsilon)
		# init. task cons as a nn param
		# am using variance scld avg of task effect map from SPM as init here.
        # in future, this init will be removed.
        self.beta_init = torch.FloatTensor(np.array(nib.load(task_init).dataobj)).to(self.device)
        self.task_init = torch.nn.Parameter(self.beta_init)
        self.inducing_pts = num_inducing_pts
        self.mll_scale = torch.as_tensor((mll_scale)).to(self.device)
        # max_ls term is used to avoid ls from blowing up.
        # I used 10 here but there is not much of a difference in output for reasonable values
        self.max_ls = torch.as_tensor(5.0).to(self.device)
		#init params for GPs
		#these are Xus (not trainable), Yu's, ls and kvar (trainable)
		#pass these to a big dict -- i.e., gp_params
        self.gp_params  = {'task':{}, 'x':{}, 'y':{}, 'z':{}, 'xrot':{}, 'yrot':{}, 'zrot':{}}
        #set prior for Yu's
        #will init Yu's by sampling from this dist.
        #so ==10 is arbitrary -- other vals might work too
        pu_prior = MultivariateNormal(torch.zeros(self.inducing_pts), 10*torch.eye(self.inducing_pts))
        #for task
        #add mean and variance of linear weight as trainable params
        #start with sa==0, va==1
        #that is, our prior is N(0, 1)
        #no params for actual GP --> this is a binary variable!!!
        self.sa_task = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['task']['sa'] = self.sa_task
        self.va_task = torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
        self.gp_params['task']['va'] = self.va_task

		#Now init GP + lin params for other regressors (all of these are NOT binary)
        #Ranges used are the ones for z-scored inputs.
		#x trans
        self.xu_x = torch.linspace(-4.00, 3.50, self.inducing_pts).to(self.device)
        self.gp_params['x']['xu'] = self.xu_x
        self.Y_x = torch.nn.Parameter(pu_prior.rsample().to(self.device))
        self.gp_params['x']['y'] = self.Y_x
        self.logkvar_x = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['x']['logkvar'] = self.logkvar_x
        self.logls_x = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['x']['log_ls'] = self.logls_x
        self.sa_x = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['x']['sa'] = self.sa_x
        self.va_x = torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
        self.gp_params['x']['va'] = self.va_x
        #y trans
        self.xu_y = torch.linspace(-2.5, 3.42, self.inducing_pts).to(self.device)
        self.gp_params['y']['xu'] = self.xu_y
        self.Y_y = torch.nn.Parameter(pu_prior.rsample().to(self.device))
        self.gp_params['y']['y'] = self.Y_y
        self.logkvar_y = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['y']['logkvar'] = self.logkvar_y
        self.logls_y = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['y']['log_ls'] = self.logls_y
        self.sa_y = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['y']['sa'] = self.sa_y
        self.va_y = torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
        self.gp_params['y']['va'] = self.va_y
        #z trans
        self.xu_z = torch.linspace(-3.45, 3.80, self.inducing_pts).to(self.device)
        self.gp_params['z']['xu'] = self.xu_z
        self.Y_z = torch.nn.Parameter(pu_prior.rsample().to(self.device))
        self.gp_params['z']['y'] = self.Y_z
        self.logkvar_z = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['z']['logkvar'] = self.logkvar_z
        self.logls_z = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['z']['log_ls'] = self.logls_z
        self.sa_z = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['z']['sa'] = self.sa_z
        self.va_z = torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
        self.gp_params['z']['va'] = self.va_z

        #rotational ones
        #xrot
        self.xu_xrot = torch.linspace(-3.31, 2.73, self.inducing_pts).to(self.device)
        self.gp_params['xrot']['xu'] = self.xu_xrot
        self.Y_xrot = torch.nn.Parameter(pu_prior.rsample().to(self.device))
        self.gp_params['xrot']['y'] = self.Y_xrot
        self.logkvar_xrot= torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['xrot']['logkvar'] = self.logkvar_xrot
        self.logls_xrot= torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['xrot']['log_ls'] = self.logls_xrot
        self.sa_xrot = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['xrot']['sa'] = self.sa_xrot
        self.va_xrot = torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
        self.gp_params['xrot']['va'] = self.va_xrot

        #yrot
        self.xu_yrot = torch.linspace(-3.14, 4.76, self.inducing_pts).to(self.device)
        self.gp_params['yrot']['xu'] = self.xu_yrot
        self.Y_yrot= torch.nn.Parameter(pu_prior.rsample().to(self.device))
        self.gp_params['yrot']['y'] = self.Y_yrot
        self.logkvar_yrot = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['yrot']['logkvar'] = self.logkvar_yrot
        self.logls_yrot= torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['yrot']['log_ls'] = self.logls_yrot
        self.sa_yrot = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['yrot']['sa'] = self.sa_yrot
        self.va_yrot = torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
        self.gp_params['yrot']['va'] = self.va_yrot

        #zrot
        self.xu_zrot = torch.linspace(-3.03, 2.54, self.inducing_pts).to(self.device)
        self.gp_params['zrot']['xu'] = self.xu_zrot
        self.Y_zrot= torch.nn.Parameter(pu_prior.rsample().to(self.device))
        self.gp_params['zrot']['y'] = self.Y_zrot
        self.logkvar_zrot= torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['zrot']['logkvar'] = self.logkvar_zrot
        self.logls_zrot= torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['zrot']['log_ls'] = self.logls_zrot
        self.sa_zrot = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['zrot']['sa'] = self.sa_zrot
        self.va_zrot = torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
        self.gp_params['zrot']['va'] = self.va_zrot
        # init z_prior --> for VAE latents
        # When initializing mean, cov_factor and cov_diag '.to(self.device)'
        #piece is  NEEDED for vals to be  properly passed to CUDA.
        mean = torch.zeros(self.num_latents).to(self.device)
        cov_factor = torch.zeros(self.num_latents).unsqueeze(-1).to(self.device)
        cov_diag = torch.ones(self.num_latents).to(self.device)
        self.z_prior = LowRankMultivariateNormal(mean, cov_factor, cov_diag)
        self._build_network()
        self.optimizer = Adam(self.parameters(), lr=self.lr)
        self.epoch = 0
        self.loss = {'train':{}, 'test':{}}
        self.to(self.device)

    def _build_network(self):
        # added track_running_stats = False flag to batch_norm _get_layers
        # this improves behavior during test loss calc
        # Encoder
        self.conv1 = nn.Conv3d(1,self.nf,3,1)
        self.conv2 = nn.Conv3d(self.nf,self.nf,3,2)
        self.conv3 = nn.Conv3d(self.nf,2*self.nf,3,1)
        self.conv4 = nn.Conv3d(2*self.nf,2*self.nf,3,2)
        self.conv5 = nn.Conv3d(2*self.nf,2*self.nf,3,1)
        self.bn1 = nn.BatchNorm3d(1, track_running_stats=False)
        self.bn3 = nn.BatchNorm3d(self.nf, track_running_stats=False)
        self.bn5 = nn.BatchNorm3d(2*self.nf, track_running_stats=False)
        self.fc1 = nn.Linear(2*self.nf*6*8*4, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc31 = nn.Linear(100, 50)
        self.fc32 = nn.Linear(100, 50)
        self.fc33 = nn.Linear(100, 50)
        self.fc41 = nn.Linear(50, self.num_latents)
        self.fc42 = nn.Linear(50, self.num_latents)
        self.fc43 = nn.Linear(50, self.num_latents)

        #Decoder
        self.fc5 = nn.Linear(self.z_dim, 50) # z_dim would be z+k+1. Here should be 40 - 32 z's + 7 covariate + base.
        self.fc6 = nn.Linear(50, 100)
        self.fc7 = nn.Linear(100, 200)
        self.fc8 = nn.Linear(200, 2*self.nf*6*8*5)
        self.convt1 = nn.ConvTranspose3d(2*self.nf,2*self.nf,3,1)
        self.convt2 = nn.ConvTranspose3d(2*self.nf,2*self.nf,3,2, padding=(1,0,1), output_padding=(1,0,1))
        self.convt3 = nn.ConvTranspose3d(2*self.nf,self.nf,3,1)
        self.convt4 = nn.ConvTranspose3d(self.nf,self.nf,(5,3,3),2)
        self.convt5 = nn.ConvTranspose3d(self.nf,1,3,1)
        self.bnt1 = nn.BatchNorm3d(2*self.nf, track_running_stats=False)
        self.bnt3 = nn.BatchNorm3d(2*self.nf, track_running_stats=False)
        self.bnt5 = nn.BatchNorm3d(self.nf, track_running_stats=False)

    def _get_layers(self):
        """Return a dictionary mapping names to network layers.
        Again, adaptions here were minimal -- enough to match layers defined
        in __build_network.
        """
        return {'fc1':self.fc1, 'fc2':self.fc2, 'fc31':self.fc31,
                'fc32':self.fc32, 'fc33':self.fc33, 'fc41':self.fc41,
                'fc42':self.fc42, 'fc43':self.fc43, 'fc5':self.fc5,
                'fc6':self.fc6, 'fc7':self.fc7, 'fc8':self.fc8, 'bn1':self.bn1,
                'bn3':self.bn3, 'bn5':self.bn5,'bnt1':self.bnt1, 'bnt3':self.bnt3,
                'bnt5':self.bnt5, 'conv1':self.conv1,'conv2':self.conv2,
                'conv3':self.conv3, 'conv4':self.conv4,
                'conv5':self.conv5,'convt1':self.convt1, 'convt2':self.convt2,
                'convt3':self.convt3, 'convt4':self.convt4,
                'convt5':self.convt5}

    def encode(self, x):
        x = x.view(-1,1,IMG_SHAPE[0],IMG_SHAPE[1],IMG_SHAPE[2])
        h = F.relu(self.conv1(self.bn1(x)))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(self.bn3(h)))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(self.bn5(h)))
        h = h.view(-1,2*self.nf*6*8*4)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        mu = F.relu(self.fc31(h))
        mu = self.fc41(mu)
        u = F.relu(self.fc32(h))
        u = self.fc42(u).unsqueeze(-1) # Last dimension is rank of \Sigma = 1.
        d = F.relu(self.fc33(h))
        d = torch.exp(self.fc43(d)) # d must be positive.
        return mu, u, d

    def decode(self, z):
        h = F.relu(self.fc5(z))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = F.relu(self.fc8(h))
        h = h.view(-1,2*self.nf,6,8,5)
        h = F.relu(self.convt1(self.bnt1(h)))
        h = F.relu(self.convt2(h))
        h = F.relu(self.convt3(self.bnt3(h)))
        h = F.relu(self.convt4(h))
        return torch.sigmoid(self.convt5(self.bnt5(h)).squeeze(1).view(-1,IMG_DIM))

    #modules added to substitute too similar/redundant GP min-batch samples
    #these were causing issues with cholesky decompositions in GP module

    def get_neighbors_dict(self, xqs):
        """
        Create dict with keys being xqs indeces and
        entries being index for pt itself + for any other points
        within a given rounding threshold of it.
        """
        neighbors = {}
        xqs = xqs.detach().cpu().numpy()
        for i in range(xqs.shape[0]):
            item = xqs[i]
            item_nn_idxs = [i] #add item itself to its nn list
            for j in range(xqs.shape[0]):
                diff = np.abs(item - xqs[j])
                if i!=j and diff <= 1e-3:
                    item_nn_idxs.append(j)
            neighbors[str(i)] = item_nn_idxs
        return neighbors

    def get_curated_samples(self, neighbors_dict, xqs):
        """
        Uses neighbors_dict to create another dict containing:
        1) Curated Xq entries to be passed to GP.
        This is either pt itself if entry has no neighbors OR
        average of avaible neighbor points.
        2) Number of neighbors for each unique entry. This will be used
        to scale Sigma_0.
        3) Unique_idxs: indeces for unique entries/pts
        4) Deleted_idxs: indeces for entries that were considered redundant.
        These are incorporated into average for a unique pt.
        """
        xqs = xqs.detach().cpu().numpy()
        means, nn_counts = [], []
        unique_idxs, deleted_idxs, idxs_accounted_for = [], [], []
        for i in range(len(list(neighbors_dict.keys()))):
            if i not in idxs_accounted_for:
                curr_nn = neighbors_dict[str(i)]
                unique_idxs.append(i)
                if len(curr_nn)==1:
                    #pt has no neighbors
                    means.append(xqs[curr_nn[0]])
                    nn_counts.append(1.0)
                    idxs_accounted_for.append(i)
                else:
                    #pt has neighbors
                    means.append(np.mean(np.take(xqs, curr_nn)))
                    nn_counts.append(len(curr_nn))
                    idxs_accounted_for.extend(curr_nn)
            else:
                deleted_idxs.append(i)

        curated_samples = {'mean': np.array(means), 'nn_counts': np.array(nn_counts), \
        'unique_idxs': np.array(unique_idxs), 'deleted_idxs':np.array(deleted_idxs)}
        return curated_samples


    def get_replacement_sample(self, idx_replaced, neighbors_dict, unique_idxs, samples):
        """
        Gets replacement sample for a redundant input.
        Essentially uses sample for corresponding unique component of
        MV Gaussian as surrogate sample for the replaced/redundant pt.
        """
        replacement_set = neighbors_dict[str(idx_replaced)]
        for i in replacement_set:
            if i in unique_idxs:
                replacement_sample = samples[np.where(unique_idxs==i)]
                return replacement_sample

    def get_all_samples(self, neighbors_dict, idxs_replaced, unique_idxs, samples):
        """
        Constructs y_q sample tensor.
        For unique pts/entries --> samples are obtained using GP module's rsample() method.
        For non-unique/replaced pts --> samples are obtained using get_replacement_sample
        method.
        """
        samples = samples.detach().cpu().numpy()
        #get array w/ all replacement samples
        replaced_samples = []
        for i in idxs_replaced:
            replacement = self.get_replacement_sample(i, neighbors_dict, unique_idxs, samples)
            replaced_samples.append(replacement)
        replaced_samples = np.array(replaced_samples)
        #now create all samples arr
        all_samples = np.zeros(len(list(neighbors_dict.keys())))
        np.put(all_samples, idxs_replaced.astype(int), replaced_samples)
        np.put(all_samples, unique_idxs.astype(int), samples)
        return torch.FloatTensor(all_samples).to(self.device)


    def forward(self, ids, covariates, x, return_latent_rec=False):
        imgs = {'base': {}, 'task': {}, 'x_mot':{}, 'y_mot':{},'z_mot':{}, 'pitch_mot':{},\
        'roll_mot':{}, 'yaw_mot':{},'full_rec': {}}
        imgs_keys = list(imgs.keys())
        gp_params_keys = list(self.gp_params.keys())
        #set batch GP_loss to zero
        gp_loss = 0
        #set L1 regularization term to zero
        l1_reg = 0
        #getting z's using encoder
        mu, u, d = self.encode(x)
        #check if d is not too small
        #if d is too small, add a small # before using it
        #this solves some issues I was having with nan losses
        d_small = d[d<1e-6]
        if len(d_small)>= 1:
            d = d.add(1e-6)
        latent_dist = LowRankMultivariateNormal(mu, u, d)
        z = latent_dist.rsample()
        base_oh = torch.nn.functional.one_hot(torch.zeros(ids.shape[0],\
        dtype=torch.int64), self.num_covariates+1)
        base_oh = base_oh.to(self.device).float()
        zcat = torch.cat([z, base_oh], 1).float()
        x_rec = self.decode(zcat).view(x.shape[0], -1)
        imgs['base'] = x_rec.detach().cpu().numpy()
        for i in range(1,self.num_covariates+1):
            #print(i)
            cov_oh = torch.nn.functional.one_hot(i*torch.ones(ids.shape[0],\
            dtype=torch.int64), self.num_covariates+1)
            cov_oh = cov_oh.to(self.device).float()
            zcat = torch.cat([z, cov_oh], 1).float()
            diff = self.decode(zcat).view(x.shape[0], -1)
            #get xqs, these are inputs for query pts
            xq = covariates[:, i-1]
            #get linear weights
            ma = Normal(self.gp_params[gp_params_keys[i-1]]['sa'], self.gp_params[gp_params_keys[i-1]]['va'])
            ka = ma.rsample()
            #now compute lin term kl
            #and add it to GP loss
            gp_linW_kl = calc_linW_KL(self.gp_params[gp_params_keys[i-1]]['sa'], \
            self.gp_params[gp_params_keys[i-1]]['va'])
            gp_loss += gp_linW_kl
            #set non-linear (GP term) to zeros
            #this will remain zero for bin covariates... And will be updated for continuous ones.
            y_q = torch.zeros(ids.shape[0]).to(self.device)
            if i!=1:
                #get params for GP regressor
                Xu = self.gp_params[gp_params_keys[i-1]]['xu']
                Yu = self.gp_params[gp_params_keys[i-1]]['y']
                #mk sure kvar is at least some small positive number
                kvar = (self.gp_params[gp_params_keys[i-1]]['logkvar']).exp() + 0.1
                #mk sure ls doesn't grow to infinity
                sig = nn.Sigmoid()
                ls = self.max_ls * sig((self.gp_params[gp_params_keys[i-1]]['log_ls']).exp() + 0.5)
                #get curated xqs
                neighbors_dict = self.get_neighbors_dict(xq)
                curated_samples = self.get_curated_samples(neighbors_dict, xq)
                #creat GP obj
                gp_regressor = gp.GP(Xu, Yu, kvar, ls)
                #get samples for unique pts
                unique_samples = gp_regressor.rsample(torch.FloatTensor(curated_samples['mean']).to(self.device), i, self.save_dir)
                #get full sample set
                y_q = self.get_all_samples(neighbors_dict, curated_samples['deleted_idxs'], \
                curated_samples['unique_idxs'], unique_samples)
                #calc actual GP kl and add it to GP loss
                gp_kl = gp_regressor.calc_GP_kl(torch.FloatTensor(curated_samples['mean']).to(self.device))
                gp_loss += gp_kl
            #use linear weight + GP prediction to construct full scalling factor
            task_var = (ka * covariates[:, i-1]) + y_q
            #convolve FULL scaling factor w/ HRF
            #this is done for biological regressors only
            ##will need to  make a cleaner version of the implementation below
            if i ==1:
                #if covariate is task...
                #need to convolve (GP output + k * covariate) result with HRF
                tr_times = np.arange(0, 20, 1.4) #TR==1.4 here
                hrf_at_trs = hrf(tr_times)
                time_series = np.convolve(task_var.detach().cpu().numpy(), hrf_at_trs)
                n_to_remove = len(hrf_at_trs) - 1
                time_series = time_series[:-n_to_remove]
                task_var = torch.FloatTensor(time_series).to(self.device)
            #use this to scale effect map
            #using EinSum to preserve batch dim
            cons = torch.einsum('b,bx->bx', task_var, diff)
            #add cons to init_task param if covariate == 'task'
            #implementation below was adopted to avoid in place ops that would cause autograd errors
            #in future, this init piece will go away entirely
            if i==1:
                cons = cons + self.task_init.unsqueeze(0).contiguous().view(1, -1).expand(ids.shape[0], -1)
            # am forcing l1 regularization on all maps (including motion ones)
            l1_loss = torch.norm(cons, p=1)
            l1_reg += l1_loss
            x_rec = x_rec + cons
            imgs[imgs_keys[i]] = cons.detach().cpu().numpy()
        imgs['full_rec']=x_rec.detach().cpu().numpy()
        # calculating loss for VAE ...
        elbo = -kl.kl_divergence(latent_dist, self.z_prior) #shape = batch_dim
        #obs_dist.shape = batch_dim, img_dim
        obs_dist = Normal(x_rec.float(),\
        torch.exp(-self.epsilon.unsqueeze(0).view(1, -1).expand(ids.shape[0], -1)).float())
        log_prob = obs_dist.log_prob(x.view(ids.shape[0], -1))
        #sum over img_dim to get a batch_dim tensor
        sum_log_prob = torch.sum(log_prob, dim=1)
        elbo = elbo + sum_log_prob
        #contract all values using torch.mean()
        elbo = torch.mean(elbo, dim=0)
        #adding GP losses to VAE loss
        tot_loss = -elbo + self.mll_scale*(gp_loss) + self.l1_scale*(l1_reg) #swap mll_scale for gp_scale
        if return_latent_rec:
            return tot_loss, z.detach().cpu().numpy(), imgs
        return tot_loss

    #on train method below, I commented autograd.detect.anomaly() line
    #which was used to trace out nan loss issue
    #only use this if trying to trace issues with auto-grad
    #otherwise, it will significantly slow code execution!
    def train_epoch(self, train_loader):
        self.train()
        train_loss = 0.0
        #with autograd.detect_anomaly():
        for batch_idx, sample in enumerate(train_loader):
            x = sample['volume']
            x = x.to(self.device)
            covariates = sample['covariates']
            covariates = covariates.to(self.device)
            ids = sample['subjid']
            ids = ids.to(self.device)
            loss = self.forward(ids, covariates, x)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        train_loss /= len(train_loader.dataset)
        print('Epoch: {} Average loss: {:.4f}'.format(self.epoch, train_loss))
        self.epoch += 1
        return train_loss

    def test_epoch(self, test_loader):
        self.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                x = sample['volume']
                x = x.to(self.device)
                covariates = sample['covariates']
                covariates = covariates.to(self.device)
                ids = sample['subjid']
                ids = ids.to(self.device)
                loss = self.forward(ids, covariates, x)
                test_loss += loss.item()
        test_loss /= len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    def save_state(self, filename):
        layers = self._get_layers()
        state = {}
        for layer_name in layers:
            state[layer_name] = layers[layer_name].state_dict()
        state['optimizer_state'] = self.optimizer.state_dict()
        state['loss'] = self.loss
        state['z_dim'] = self.z_dim
        state['epoch'] = self.epoch
        state['lr'] = self.lr
        state['save_dir'] = self.save_dir
        state['epsilon'] = self.epsilon
        state['task_init'] = self.task_init
        state['beta_init'] = self.beta_init
        state['l1_scale'] = self.l1_scale
        #add GP nn params to checkpt files
        #this includes gp_params dict
        state['sa_task'] = self.sa_task
        state['va_task'] = self.va_task
        state['Y_x'] = self.Y_x
        state['logkvar_x'] = self.logkvar_x
        state['logls_x'] = self.logls_x
        state['sa_x'] = self.sa_x
        state['va_x'] = self.va_x
        state['Y_y'] = self.Y_y
        state['logkvar_y'] = self.logkvar_y
        state['logls_y'] = self.logls_y
        state['sa_y'] = self.sa_y
        state['va_y'] = self.va_y
        state['Y_z'] = self.Y_z
        state['logkvar_z'] = self.logkvar_z
        state['logls_z'] = self.logls_z
        state['sa_z'] = self.sa_z
        state['va_z'] = self.va_z
        state['Y_xrot'] = self.Y_xrot
        state['logkvar_xrot'] = self.logkvar_xrot
        state['logls_xrot'] = self.logls_xrot
        state['sa_xrot'] = self.sa_xrot
        state['va_xrot'] = self.va_xrot
        state['Y_yrot'] = self.Y_yrot
        state['logkvar_yrot'] = self.logkvar_yrot
        state['logls_yrot'] = self.logls_yrot
        state['sa_yrot'] = self.sa_yrot
        state['va_yrot'] = self.va_yrot
        state['Y_zrot'] = self.Y_zrot
        state['logkvar_zrot'] = self.logkvar_zrot
        state['logls_zrot'] = self.logls_zrot
        state['sa_zrot'] = self.sa_zrot
        state['va_zrot'] = self.va_zrot
        state['gp_params'] = self.gp_params
        state['mll_scale'] = self.mll_scale
        state['inducing_pts'] = self.inducing_pts
        filename = os.path.join(self.save_dir, filename)
        torch.save(state, filename)

    def load_state(self, filename):
        checkpoint = torch.load(filename)
        assert checkpoint['z_dim'] == self.z_dim
        layers = self._get_layers()
        for layer_name in layers:
            layer = layers[layer_name]
            layer.load_state_dict(checkpoint[layer_name])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.loss = checkpoint['loss']
        self.epoch = checkpoint['epoch']
        self.epsilon = checkpoint['epsilon']
        self.beta_init = checkpoint['beta_init']
        self.task_init = checkpoint['task_init']
        self.l1_scale = checkpoint['l1_scale']
        #load in GP params from ckpt files
        #and load gp_params dict - needed here (otherwise only initial values are plotted!)
        self.sa_task = checkpoint['sa_task']
        self.va_task = checkpoint['va_task']
        self.Y_x = checkpoint['Y_x']
        self.logkvar_x = checkpoint['logkvar_x']
        self.logls_x = checkpoint['logls_x']
        self.sa_x = checkpoint['sa_x']
        self.va_x = checkpoint['va_x']
        self.Y_y = checkpoint['Y_y']
        self.logkvar_y = checkpoint['logkvar_y']
        self.logls_y = checkpoint['logls_y']
        self.sa_y = checkpoint['sa_y']
        self.va_y = checkpoint['va_y']
        self.Y_z = checkpoint['Y_z']
        self.logkvar_z = checkpoint['logkvar_z']
        self.logls_z = checkpoint['logls_z']
        self.sa_z = checkpoint['sa_z']
        self.va_z = checkpoint['va_z']
        self.Y_xrot = checkpoint['Y_xrot']
        self.logkvar_xrot = checkpoint['logkvar_xrot']
        self.logls_xrot = checkpoint['logls_xrot']
        self.sa_xrot = checkpoint['sa_xrot']
        self.va_xrot = checkpoint['va_xrot']
        self.Y_yrot = checkpoint['Y_yrot']
        self.logkvar_yrot = checkpoint['logkvar_yrot']
        self.logls_yrot= checkpoint['logls_yrot']
        self.sa_yrot = checkpoint['sa_yrot']
        self.va_yrot = checkpoint['va_yrot']
        self.Y_zrot = checkpoint['Y_zrot']
        self.logkvar_zrot = checkpoint['logkvar_zrot']
        self.logls_zrot = checkpoint['logls_zrot']
        self.sa_zrot = checkpoint['sa_zrot']
        self.va_zrot = checkpoint['va_zrot']
        self.gp_params = checkpoint['gp_params']
        self.mll_scale = checkpoint['mll_scale']
        self.inducing_pts = checkpoint['inducing_pts']

    def project_latent(self, loaders_dict, save_dir, title=None, split=98):
        # plotting only test set since this is un-shuffled.
        # Will plot by subjid in future to overcome this limitation
        # Collect latent means.
        filename = str(self.epoch).zfill(3) + '_temp.pdf'
        file_path = os.path.join(save_dir, filename)
        latent = np.zeros((len(loaders_dict['test'].dataset), self.num_latents))
        with torch.no_grad():
            j = 0
            for i, sample in enumerate(loaders_dict['test']):
                x = sample['volume']
                x = x.to(self.device)
                mu, _, _ = self.encode(x)
                latent[j:j+len(mu)] = mu.detach().cpu().numpy()
                j += len(mu)
        # UMAP them.
        transform = UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
        metric='euclidean', random_state=42)
        projection = transform.fit_transform(latent)
        #print(projection.shape)
        # Plot.
        c_list = ['b','g','r','c','m','y','k','orange','blueviolet','hotpink',\
        'lime','skyblue','teal','sienna']
        colors = itertools.cycle(c_list)
        data_chunks = range(0,len(loaders_dict['test'].dataset),split)  #check value of split here
        #print(data_chunks)
        for i in data_chunks:
            t = np.arange(split)
            plt.scatter(projection[i:i+split,0], projection[i:i+split,1],\
            color=next(colors), s=1.0, alpha=0.6)
            #commenting plot by time
            #plt.scatter(projection[i:i+split,0], projection[i:i+split,1], c=t, s=1.0, alpha=0.6)
            plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.savefig(file_path)
        #Uncomment this if we actually wish to get latent and projections
        #return latent, projection

    def reconstruct(self, item, ref_nii, save_dir):
        """Reconstruct a volume and its cons given a dset idx."""
        x = item['volume'].unsqueeze(0)
        x = x.to(self.device)
        covariates = item['covariates'].unsqueeze(0)
        covariates = covariates.to(self.device)
        ids = item['subjid'].view(1)
        ids = ids.to(self.device)
        with torch.no_grad():
            _, _, imgs = self.forward(ids, covariates, x, return_latent_rec = True)
            for key in imgs.keys():
                filename = 'recon_{}.nii'.format(key)
                filepath = os.path.join(save_dir, filename)
                reconstructed = imgs[key]
                recon_array = reconstructed.reshape(41,49,35)
                #use nibabel to load in header and affine of filename
                #call that when writing recon_nifti
                input_nifti = nib.load(ref_nii)
                recon_nifti = nib.Nifti1Image(recon_array, input_nifti.affine, input_nifti.header)
                nib.save(recon_nifti, filepath)

    def plot_GPs(self, csv_file = '', save_dir = ''):
        """
        Plot inducing points &
        posterior mean +/- 2tds for a trained GPs
        Also outputs : 1) a file containing per covariate GP mean variance
        This info is used by post-processing scripts to merge maps
        of cte covariates with base map. 2) Several csv files (one per covariate)
        with sorted xqs and their corresponding predicted means and variances.

        Parameters
        ---------
        csv_file: file containing data for model
        this is the same used for data laoders
        """
        #create dict to hold covariate yq variances
        keys = ['x_mot', 'y_mot', 'z_mot', 'pitch_mot', 'roll_mot', 'yaw_mot']
        covariates_mean_vars = dict.fromkeys(keys)
        #setup output dir
        outdir_name = str(self.epoch).zfill(3) + '_GP_plots'
        plot_dir = os.path.join(save_dir, outdir_name)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        #read in values for each regressor from csv_file
        #pass them to torch as a float tensor
        data = pd.read_csv(csv_file)
        all_covariates = data[['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z']]
        all_covariates = all_covariates.to_numpy()
        all_covariates = torch.from_numpy(all_covariates)
        regressors = list(self.gp_params.keys())
        for i in range(1, len(regressors)): #skip task
            #create dict to hold all entries for each cov -- original input + predicted mean
            #and predicted vars for all points
            curr_cov = {};
            #build GP for the regressor
            xu = self.gp_params[regressors[i]]['xu']
            yu = self.gp_params[regressors[i]]['y']
            kvar = (self.gp_params[regressors[i]]['logkvar']).exp() + 0.1
            sig = nn.Sigmoid()
            ls = self.max_ls * sig((self.gp_params[regressors[i]]['log_ls']).exp() + 0.5)
            gp_regressor = gp.GP(xu, yu, kvar, ls)
            #get all xi's for regressor
            #get prediction
            covariates = all_covariates[:, i-1]
            xq = covariates.to(self.device)
            yq, yvar = gp_regressor.predict(xq)
            #add vals to covar dict
            curr_cov["xq"] = covariates
            curr_cov["mean"] = yq.detach().cpu().numpy().tolist()
            curr_cov["vars"] = yvar.detach().cpu().numpy().tolist()
            #save this dict
            outfull_name = str(self.epoch).zfill(3) + '_GP_' + keys[i-1] + '_full.csv'
            covariate_full_data = pd.DataFrame.from_dict(curr_cov)
            #sort out predictions
            sorted_full_data = covariate_full_data.sort_values(by=["xq"])
            #save them to csv file just in case
            sorted_full_data.to_csv(os.path.join(plot_dir, outfull_name))
            #calc variance of predicted GP mean
            #and pass it to dict
            yq_variance = torch.var(yq)
            covariates_mean_vars[keys[i-1]] = [yq_variance.detach().cpu().numpy()]
            #pass vars to cpu and np prior to plotting
            x_u = xu.detach().cpu().numpy()
            y_u = yu.detach().cpu().numpy()
            #create plots and save them
            plt.clf()
            plt.plot(sorted_full_data["xq"], sorted_full_data["mean"], c='darkblue', alpha=0.5, label='posterior mean')
            two_sigma = 2*np.sqrt(sorted_full_data["vars"])
            kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
            plt.fill_between(sorted_full_data["xq"], (sorted_full_data["mean"]-two_sigma), (sorted_full_data["mean"]+two_sigma), **kwargs)
            plt.scatter(x_u, y_u, c='k', label='inducing points')
            plt.locator_params(axis='x', nbins = 6)
            plt.locator_params(axis='y', nbins = 4)
            plt.legend(loc='best')
            plt.title('GP Plot {}_{}'.format(regressors[i], 'full_set'))
            plt.xlabel('Covariate')
            plt.ylabel('GP Prediction')
            #save plot
            filename = 'GP_{}_{}.pdf'.format(regressors[i], 'full_set')
            file_path = os.path.join(plot_dir, filename)
            plt.savefig(file_path)
        #now save dict entries to a csv_file
        outcsv_name = str(self.epoch).zfill(3) + '_GP_yq_variances.csv'
        covariate_mean_vars_data = pd.DataFrame.from_dict(covariates_mean_vars)
        covariate_mean_vars_data.to_csv(os.path.join(plot_dir, outcsv_name))

    def train_loop(self, loaders, epochs=100, test_freq=2, save_freq=10, save_dir = ''):
        print("="*40)
        print("Training: epochs", self.epoch, "to", self.epoch+epochs-1)
        print("Training set:", len(loaders['train'].dataset))
        print("Test set:", len(loaders['test'].dataset))
        print("="*40)
        # For some number of epochs...
        for epoch in range(self.epoch, self.epoch+epochs):
            #Run through the training data and record a loss.
            loss = self.train_epoch(loaders['train'])
            self.loss['train'][epoch] = loss
            # Run through the test data and record a loss.
            if (test_freq is not None) and (epoch % test_freq == 0):
                loss = self.test_epoch(loaders['test'])
                self.loss['test'][epoch] = loss
            # Save the model.
            if (save_freq is not None) and (epoch % save_freq == 0) and (epoch > 0):
                filename = "checkpoint_"+str(epoch).zfill(3)+'.tar'
                file_path = os.path.join(save_dir, filename)
                self.save_state(file_path)

if __name__ == "__main__":
    pass
