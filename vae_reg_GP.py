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
from torch.utils.tensorboard import SummaryWriter
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
import datetime
from scipy.stats import norm
from scipy import ndimage

#OLD HRF
def hrf1(times):
    """ Return values for HRF at given times """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

#HRF (as implemented in FSL)
#only diff here is scaling to max 0.6
#sometime of scaling will likely be needed here -- for stability purposes
def hrf2(times):
    """ Return values for HRF at given times """
    peak_values = gamma.pdf(times, 6, scale=(2.449*2.449))
    undershoot_values = gamma.pdf(times, 16, scale=(4*4))
    values = peak_values - 0.167*undershoot_values
    #once again, scale max to 0.6
    return values / np.max(values)*0.6

# maintained shape of original nn by downsampling data on preprocessing
IMG_SHAPE = (41,49,35)
IMG_DIM = np.prod(IMG_SHAPE)

class VAE(nn.Module):
    def __init__(self, nf=8, save_dir='', lr=1e-3, num_covariates=7, num_latents=32, device_name="auto", \
    num_inducing_pts=6, mll_scale=10.0, l1_scale=1.0, glm_maps = '', glm_reg_scale=1):
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
            self.device = torch.device("cuda:0")
        if self.save_dir != '' and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
		# init epsilon param modeling single voxel variance
		# -log(10) initial value accounts for removing model_precision term from original version
        epsilon = -np.log(10)*torch.ones([IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]], dtype=torch.float64, device = self.device)
        self.epsilon = torch.nn.Parameter(epsilon)
        #read in GLM maps for regularizer
        glm_maps = pd.read_csv(glm_maps).to_numpy()
        self.glm_maps = torch.from_numpy(glm_maps).to(self.device) #shape is 70, 315x8
        self.glm_reg_scale = glm_reg_scale
        self.inducing_pts = num_inducing_pts
        self.mll_scale = torch.as_tensor((mll_scale)).to(self.device)
        # max_ls term is used to avoid ls from blowing up.
        self.max_ls = torch.as_tensor(3.0).to(self.device)
        #set so_sqrd term for prior over inducing points
        self.so_sqrd = torch.as_tensor(10.0).to(self.device)
		#init params for GPs
        self.gp_params  = {'task':{}, 'x':{}, 'y':{}, 'z':{}, 'xrot':{}, 'yrot':{}, 'zrot':{}}
        #for task
        #initialize params representing kappa posterior mean (sa) and std (va)
        #no params for non-linear GP piece --> this is a binary variable!!!
        self.sa_task = torch.nn.Parameter(torch.normal(1, 1, size=(1,1)).to(self.device))
        self.gp_params['task']['sa'] = self.sa_task
        self.va_task = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['task']['va'] = self.va_task

		#Now init kappa + non-linear GP params for other (non-binary) regressors
		#x trans
        self.xu_x = torch.linspace(-1.62, 5.27, self.inducing_pts).to(self.device)
        self.gp_params['x']['xu'] = self.xu_x
        self.qu_m_x = torch.nn.Parameter(torch.normal(0.0, 1.0, size=[1, self.inducing_pts]).to(self.device))
        self.gp_params['x']['qu_m'] = self.qu_m_x
        self.qu_S_x = torch.nn.Parameter(2*torch.eye(self.inducing_pts).to(self.device))
        self.gp_params['x']['qu_S'] = self.qu_S_x
        self.logkvar_x = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['x']['logkvar'] = self.logkvar_x
        self.logls_x = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['x']['log_ls'] = self.logls_x
        self.sa_x = torch.nn.Parameter(torch.normal(1, 1, size=(1,1)).to(self.device))
        self.gp_params['x']['sa'] = self.sa_x
        self.va_x = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['x']['va'] = self.va_x
        #y trans
        self.xu_y = torch.linspace(-2.98, 3.78, self.inducing_pts).to(self.device)
        self.gp_params['y']['xu'] = self.xu_y
        self.qu_m_y = torch.nn.Parameter(torch.normal(0.0, 1.0, size=[1, self.inducing_pts]).to(self.device))
        self.gp_params['y']['qu_m'] = self.qu_m_y
        self.qu_S_y = torch.nn.Parameter(2*torch.eye(self.inducing_pts).to(self.device))
        self.gp_params['y']['qu_S'] = self.qu_S_y
        self.logkvar_y = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['y']['logkvar'] = self.logkvar_y
        self.logls_y = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['y']['log_ls'] = self.logls_y
        self.sa_y = torch.nn.Parameter(torch.normal(1, 1, size=(1,1)).to(self.device))
        self.gp_params['y']['sa'] = self.sa_y
        self.va_y = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['y']['va'] = self.va_y
        #z trans
        self.xu_z = torch.linspace(-3.715, 6.56, self.inducing_pts).to(self.device)
        self.gp_params['z']['xu'] = self.xu_z
        self.qu_m_z = torch.nn.Parameter(torch.normal(0.0, 1.0, size=[1, self.inducing_pts]).to(self.device))
        self.gp_params['z']['qu_m'] = self.qu_m_z
        self.qu_S_z = torch.nn.Parameter(2*torch.eye(self.inducing_pts).to(self.device))
        self.gp_params['z']['qu_S'] = self.qu_S_z
        self.logkvar_z = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['z']['logkvar'] = self.logkvar_z
        self.logls_z = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['z']['log_ls'] = self.logls_z
        self.sa_z = torch.nn.Parameter(torch.normal(1, 1, size=(1,1)).to(self.device))
        self.gp_params['z']['sa'] = self.sa_z
        self.va_z = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['z']['va'] = self.va_z
        #rotational ones
        #xrot
        self.xu_xrot = torch.linspace(-3.50, 4.61, self.inducing_pts).to(self.device)
        self.gp_params['xrot']['xu'] = self.xu_xrot
        self.qu_m_xrot = torch.nn.Parameter(torch.normal(0.0, 1.0, size=[1, self.inducing_pts]).to(self.device))
        self.gp_params['xrot']['qu_m'] = self.qu_m_xrot
        self.qu_S_xrot = torch.nn.Parameter(2*torch.eye(self.inducing_pts).to(self.device))
        self.gp_params['xrot']['qu_S'] = self.qu_S_xrot
        self.logkvar_xrot= torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['xrot']['logkvar'] = self.logkvar_xrot
        self.logls_xrot= torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['xrot']['log_ls'] = self.logls_xrot
        self.sa_xrot = torch.nn.Parameter(torch.normal(1, 1, size=(1,1)).to(self.device))
        self.gp_params['xrot']['sa'] = self.sa_xrot
        self.va_xrot = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['xrot']['va'] = self.va_xrot

        #yrot
        self.xu_yrot = torch.linspace(-2.68, 3.36, self.inducing_pts).to(self.device)
        self.gp_params['yrot']['xu'] = self.xu_yrot
        self.qu_m_yrot = torch.nn.Parameter(torch.normal(0.0, 1.0, size=[1, self.inducing_pts]).to(self.device))
        self.gp_params['yrot']['qu_m'] = self.qu_m_yrot
        self.qu_S_yrot = torch.nn.Parameter(2*torch.eye(self.inducing_pts).to(self.device))
        self.gp_params['yrot']['qu_S'] = self.qu_S_yrot
        self.logkvar_yrot = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['yrot']['logkvar'] = self.logkvar_yrot
        self.logls_yrot= torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['yrot']['log_ls'] = self.logls_yrot
        self.sa_yrot = torch.nn.Parameter(torch.normal(1, 1, size=(1,1)).to(self.device))
        self.gp_params['yrot']['sa'] = self.sa_yrot
        self.va_yrot = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['yrot']['va'] = self.va_yrot

        #zrot
        self.xu_zrot = torch.linspace(-2.60, 3.58, self.inducing_pts).to(self.device)
        self.gp_params['zrot']['xu'] = self.xu_zrot
        self.qu_m_zrot = torch.nn.Parameter(torch.normal(0.0, 1.0, size=[1, self.inducing_pts]).to(self.device))
        self.gp_params['zrot']['qu_m'] = self.qu_m_zrot
        self.qu_S_zrot = torch.nn.Parameter(2*torch.eye(self.inducing_pts).to(self.device))
        self.gp_params['zrot']['qu_S'] = self.qu_S_zrot
        self.logkvar_zrot= torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['zrot']['logkvar'] = self.logkvar_zrot
        self.logls_zrot= torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
        self.gp_params['zrot']['log_ls'] = self.logls_zrot
        self.sa_zrot = torch.nn.Parameter(torch.normal(1, 1, size=(1,1)).to(self.device))
        self.gp_params['zrot']['sa'] = self.sa_zrot
        self.va_zrot = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
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
        #init summary writer instance for TB logging.
        ts = datetime.datetime.now().date()
        self.writer = SummaryWriter(log_dir = os.path.join(self.save_dir, 'run', ts.strftime('%m_%d_%Y')))
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

    def calc_linW_KL(self, sa, va):
        """
        Computes KL for linear weight term (kappa_\alpha)
        his term is added to KL stemming from GP itself, to yield a total GP
        contribution to the VAE-GAM objective.
        Outputs:
        KL between prior (N(0,1)) and posterior (N(sa, va^2))
        """
        post_dist = Normal(sa, va)
        prior_dist = Normal(1, 0.5)
        qk_kl = kl.kl_divergence(post_dist, prior_dist)
        return qk_kl

    def do_hrf_conv(self, input, hrf_type, scale_factor=20):
        """
        Takes an input tensor, upsamples it by some scale_factor.
        Then convolves upsampled vector with HRF of a given type.
        And then returns downsampled result to be used.
        ------
        Args:
        input: torch.tensor
        This is series of 0's or 1's corresponding to binary task variable
        for each bath entry
        hrf_type: str
        Type of HRF to be used. Can be hrf1 -- old HRF implementation OR
        hrf2, which uses delays, sigmas and ratio as being the same as for
        FSL's double gamma HRF.
        scale_factor: float
        Factor by which we will upsample input. To downsample, this implementation
        uses 1/scale_factor.
        """
        #doing HRF at TR res..
        hrf_times = np.arange(0, 20, 1.4)
        if hrf_type == 'hrf1':
            hrf_signal = torch.tensor(hrf1(hrf_times)).to(self.device)
        else:
            hrf_signal = torch.tensor(hrf2(hrf_times)).to(self.device)
        #now create tensor to carry conv step
        n_time_pts = input.shape[0]
        n_hrf_times = hrf_times.shape[0]
        shifted_hrfs = torch.zeros((n_time_pts, (n_time_pts+n_hrf_times-1))).to(self.device)
        for i in range(n_time_pts):
            shifted_hrfs[i, i : i + n_hrf_times] = hrf_signal
        #carry out convolution
        convolved_signal = torch.mm(input.unsqueeze(0), shifted_hrfs)
        #take out extra vals and return tensor of appropriate shape/size
        len_to_remove = len(hrf_times) - 1
        convolved_signal = convolved_signal.squeeze(0)[:-len_to_remove]
        return convolved_signal

    def forward(self, ids, covariates, x, log_type, return_latent_rec=False, train_mode=True):
        imgs = {'base': {}, 'task': {}, 'x_mot':{}, 'y_mot':{},'z_mot':{}, 'pitch_mot':{},\
        'roll_mot':{}, 'yaw_mot':{},'full_rec': {}}
        imgs_keys = list(imgs.keys())
        gp_params_keys = list(self.gp_params.keys())
        #set batch GP_loss to zero
        gp_loss = 0
        #set L1 regularization term to zero
        l1_reg = 0
        #set glm regularizer term to zero
        glm_reg = 0
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
        #log base map
        if train_mode:
            #log slices 12, 15, 18 for all batch elem
            self.log_map(imgs['base'], 12, 'base_map', ids.shape[0], log_type)
            self.log_map(imgs['base'], 15, 'base_map', ids.shape[0], log_type)
            self.log_map(imgs['base'], 18, 'base_map', ids.shape[0], log_type)
        for i in range(1, (self.num_covariates+1)):
            cov_oh = torch.nn.functional.one_hot(i*torch.ones(ids.shape[0],\
            dtype=torch.int64), self.num_covariates+1)
            cov_oh = cov_oh.to(self.device).float()
            zcat = torch.cat([z, cov_oh], 1).float()
            diff = self.decode(zcat).view(x.shape[0], -1)
            #get xqs, these are inputs for query pts
            xq = covariates[:, i-1]
            gp_linW_kl = self.calc_linW_KL(self.gp_params[gp_params_keys[i-1]]['sa'][0], \
            self.gp_params[gp_params_keys[i-1]]['va'][0].exp())
            gp_loss += gp_linW_kl
            beta_mean = self.gp_params[gp_params_keys[i-1]]['sa'][0] * xq
            beta_cov = torch.pow(self.gp_params[gp_params_keys[i-1]]['va'][0].exp(), 2)* torch.pow(xq, 2) * torch.eye(ids.shape[0]).to(self.device)
            if i!=1:
                #get params for GP regressor
                Xu = self.gp_params[gp_params_keys[i-1]]['xu']
                kvar = (self.gp_params[gp_params_keys[i-1]]['logkvar']).exp() + 0.1
                sig = nn.Sigmoid()
                ls = self.max_ls * sig((self.gp_params[gp_params_keys[i-1]]['log_ls']).exp() + 0.5)
                qu_m = self.gp_params[gp_params_keys[i-1]]['qu_m']
                qu_S = self.gp_params[gp_params_keys[i-1]]['qu_S']
                gp_regressor = gp.GP(Xu, kvar, ls, qu_m, qu_S)
                #update loc, scale for beta distribution
                f_bar, Sigma = gp_regressor.evaluate_posterior(xq)
                beta_mean += f_bar
                beta_cov += Sigma
                #now get Kl  for non-linear GP term
                gp_kl = gp_regressor.compute_GP_kl(self.so_sqrd, self.inducing_pts)
                gp_loss += gp_kl
            beta_dist = MultivariateNormal(beta_mean, (beta_cov + 1e-5*torch.eye(ids.shape[0]).to(self.device)))
            task_var = beta_dist.rsample()
            if train_mode:
                self.log_beta(xq, beta_mean, beta_cov, gp_params_keys[i-1], log_type) #add beta plot to TB
            #apply HRF conv to biological regressor
            if i ==1:
                task_var = self.do_hrf_conv(task_var, hrf_type='hrf1')
            #use this to scale effect map
            cons = torch.einsum('b,bx->bx', task_var, diff)
            if i==1 and train_mode==True:
                self.log_map(cons.detach().cpu().numpy(), 12, 'task_map', ids.shape[0], log_type)
                self.log_map(cons.detach().cpu().numpy(), 15, 'task_map', ids.shape[0], log_type)
                self.log_map(cons.detach().cpu().numpy(), 18, 'task_map', ids.shape[0], log_type)
            #force all maps to be close to their GLM approximations
            #am using l2 norm here to enfoce this.
            glm_diff = torch.sum(torch.cdist(cons, self.glm_maps[:, i].unsqueeze(0).expand(ids.shape[0], -1).float(), p=2))
            glm_reg += glm_diff
            # am keeping l1 reg too for now to impose sparsity on maps
            l1_loss = torch.norm(cons, p=1)
            l1_reg += l1_loss
            x_rec = x_rec + cons
            imgs[imgs_keys[i]] = cons.detach().cpu().numpy()
        imgs['full_rec']=x_rec.detach().cpu().numpy()
        #log full_rec to TB as well
        if train_mode:
            self.log_map(imgs['full_rec'], 12, 'full_reconstruction', ids.shape[0], log_type)
            self.log_map(imgs['full_rec'], 15, 'full_reconstruction', ids.shape[0], log_type)
            self.log_map(imgs['full_rec'], 18, 'full_reconstruction', ids.shape[0], log_type)
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
        tot_loss = -elbo + self.mll_scale*(gp_loss) + self.l1_scale*(l1_reg) + self.glm_reg_scale*(glm_reg)
        if return_latent_rec:
            return tot_loss, z.detach().cpu().numpy(), imgs
        return tot_loss

    #on train method below, I commented autograd.detect.anomaly() line
    #which was used to trace out nan loss issue
    #only use this if trying to trace issues with auto-grad
    #otherwise, it will significantly slow code execution!
    def train_epoch(self, train_loader):
        #self.print_gp_params()
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
            loss = self.forward(ids, covariates, x, 'train', train_mode=True)
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
                loss = self.forward(ids, covariates, x, 'test', train_mode=False)
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
        state['l1_scale'] = self.l1_scale
        state['so_sqrd'] = self.so_sqrd
        state['glm_reg_scale'] = self.glm_reg_scale
        #add GP nn params to checkpt files
        #this includes gp_params dict
        state['sa_task'] = self.sa_task
        state['va_task'] = self.va_task
        state['qu_m_x'] = self.qu_m_x
        state['qu_S_x'] = self.qu_S_x
        state['logkvar_x'] = self.logkvar_x
        state['logls_x'] = self.logls_x
        state['sa_x'] = self.sa_x
        state['va_x'] = self.va_x
        state['qu_m_y'] = self.qu_m_y
        state['qu_S_y'] = self.qu_S_y
        state['logkvar_y'] = self.logkvar_y
        state['logls_y'] = self.logls_y
        state['sa_y'] = self.sa_y
        state['va_y'] = self.va_y
        state['qu_m_z'] = self.qu_m_z
        state['qu_S_z'] = self.qu_S_z
        state['logkvar_z'] = self.logkvar_z
        state['logls_z'] = self.logls_z
        state['sa_z'] = self.sa_z
        state['va_z'] = self.va_z
        state['qu_m_xrot'] = self.qu_m_xrot
        state['qu_S_xrot'] = self.qu_S_xrot
        state['logkvar_xrot'] = self.logkvar_xrot
        state['logls_xrot'] = self.logls_xrot
        state['sa_xrot'] = self.sa_xrot
        state['va_xrot'] = self.va_xrot
        state['qu_m_yrot'] = self.qu_m_yrot
        state['qu_S_yrot'] = self.qu_S_yrot
        state['logkvar_yrot'] = self.logkvar_yrot
        state['logls_yrot'] = self.logls_yrot
        state['sa_yrot'] = self.sa_yrot
        state['va_yrot'] = self.va_yrot
        state['qu_m_zrot'] = self.qu_m_zrot
        state['qu_S_zrot'] = self.qu_S_zrot
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
        self.l1_scale = checkpoint['l1_scale']
        self.so_sqrd = checkpoint['so_sqrd']
        self.glm_reg_scale = checkpoint['glm_reg_scale']
        #load in GP params from ckpt files
        #and load gp_params dict - needed here (otherwise only initial values are plotted!)
        self.sa_task = checkpoint['sa_task']
        self.va_task = checkpoint['va_task']
        self.qu_m_x = checkpoint['qu_m_x']
        self.qu_S_x = checkpoint['qu_S_x']
        self.logkvar_x = checkpoint['logkvar_x']
        self.logls_x = checkpoint['logls_x']
        self.sa_x = checkpoint['sa_x']
        self.va_x = checkpoint['va_x']
        self.qu_m_y = checkpoint['qu_m_y']
        self.qu_S_y = checkpoint['qu_S_y']
        self.logkvar_y = checkpoint['logkvar_y']
        self.logls_y = checkpoint['logls_y']
        self.sa_y = checkpoint['sa_y']
        self.va_y = checkpoint['va_y']
        self.qu_m_z = checkpoint['qu_m_z']
        self.qu_S_z = checkpoint['qu_S_z']
        self.logkvar_z = checkpoint['logkvar_z']
        self.logls_z = checkpoint['logls_z']
        self.sa_z = checkpoint['sa_z']
        self.va_z = checkpoint['va_z']
        self.qu_m_xrot = checkpoint['qu_m_xrot']
        self.qu_S_xrot = checkpoint['qu_S_xrot']
        self.logkvar_xrot = checkpoint['logkvar_xrot']
        self.logls_xrot = checkpoint['logls_xrot']
        self.sa_xrot = checkpoint['sa_xrot']
        self.va_xrot = checkpoint['va_xrot']
        self.qu_m_yrot = checkpoint['qu_m_yrot']
        self.qu_S_yrot = checkpoint['qu_S_yrot']
        self.logkvar_yrot = checkpoint['logkvar_yrot']
        self.logls_yrot= checkpoint['logls_yrot']
        self.sa_yrot = checkpoint['sa_yrot']
        self.va_yrot = checkpoint['va_yrot']
        self.qu_m_zrot = checkpoint['qu_m_zrot']
        self.qu_S_zrot = checkpoint['qu_S_xrot']
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
            #task = np.concatenate((np.zeros(14), np.ones(14)))
            #task = np.tile(task, 3)
            #task = np.concatenate((task, np.zeros(14)))
            plt.scatter(projection[i:i+split,0], projection[i:i+split,1],\
            color=next(colors), s=1.0, alpha=0.6)
            #plt.scatter(projection[i:i+split,0], projection[i:i+split,1],\
            #c=task, s=1.0, alpha=0.6)
            #commenting plot by time
            #plt.scatter(projection[i:i+split,0], projection[i:i+split,1], c=t, s=1.0, alpha=0.6)
            plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.savefig(file_path)
        #Uncomment this if we actually wish to get latent and projections
        #return latent, projection

    def reconstruct(self, loader, ref_niis, save_dirs):
        """Reconstructs a batch of volumes
        ------
        Args:
        loader: torch.DataLoader
        ref_niis: list with paths to reference nifti files for each subj.
        These are used to trf numpy arrays into final nifti format maps.
        save_dir: root dir where vols will be saved to. Ends with subjid.
        """
        with torch.no_grad():
            for i, sample in enumerate(loader):
                x = sample['volume']
                x = x.to(self.device)
                covariates = sample['covariates']
                covariates = covariates.to(self.device)
                ids = sample['subjid']
                ids = ids.to(self.device)
                vol_num = sample['vol_num']
                subjidx = sample['subjid']
                _, _, imgs = self.forward(ids, covariates, x, 'reconstruction', return_latent_rec = True, train_mode=False)
                for key in imgs.keys():
                    gen_filename = 'recon_{}.nii'.format(key)
                    for i in range(ids.shape[0]):
                        curr_recon = imgs[key][i, :].reshape(41, 49, 35)
                        curr_vol = vol_num.tolist()[i]
                        curr_subjidx = subjidx.tolist()[i]
                        curr_savedir = save_dirs[curr_subjidx]
                        vol_dir = os.path.join(curr_savedir, 'vol_{}'.format(curr_vol))
                        if not os.path.exists(vol_dir):
                            os.makedirs(vol_dir)
                        filepath = os.path.join(vol_dir, gen_filename)
                        ref_nii = ref_niis[curr_subjidx]
                        input_nifti = nib.load(ref_nii)
                        recon_nifti = nib.Nifti1Image(curr_recon, input_nifti.affine, input_nifti.header)
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
        for i in range(1, len(regressors)): #skip task b/c its binary variable
            curr_cov = {};
            #build GP for the regressor
            xu = self.gp_params[regressors[i]]['xu']
            kvar = (self.gp_params[regressors[i]]['logkvar']).exp() + 0.1
            sig = nn.Sigmoid()
            ls = self.max_ls * sig((self.gp_params[regressors[i]]['log_ls']).exp() + 0.5)
            qu_m = self.gp_params[regressors[i]]['qu_m']
            qu_S = self.gp_params[regressors[i]]['qu_S']
            gp_regressor = gp.GP(xu, kvar, ls, qu_m, qu_S)
            #get all xi's for regressor
            covariates = all_covariates[:, i-1]
            xq = covariates.to(self.device)
            beta_diag = torch.pow(self.gp_params[regressors[i]]['va'][0].exp(), 2)* torch.pow(xq, 2) * torch.eye(xq.shape[0]).to(self.device)
            f_bar, Sigma = gp_regressor.evaluate_posterior(xq)
            beta_mean = (self.gp_params[regressors[i]]['sa'][0] * xq) + f_bar
            beta_cov = beta_diag + Sigma
            #add vals to covar dict
            curr_cov["xq"] = covariates
            curr_cov["mean"] = beta_mean.detach().cpu().numpy().tolist()
            curr_cov["vars"] = torch.diag(beta_cov).detach().cpu().numpy().tolist()
            #save this dict
            outfull_name = str(self.epoch).zfill(3) + '_GP_' + keys[i-1] + '_full.csv'
            covariate_full_data = pd.DataFrame.from_dict(curr_cov)
            #sort out predictions
            sorted_full_data = covariate_full_data.sort_values(by=["xq"])
            #save them to csv file just in case
            sorted_full_data.to_csv(os.path.join(plot_dir, outfull_name))
            #calc variance of predicted GP mean
            #and pass it to dict
            beta_mean_variance = torch.var(beta_mean)
            covariates_mean_vars[keys[i-1]] = [beta_mean_variance.detach().cpu().numpy()]
            #create plots and save them
            plt.clf()
            plt.plot(sorted_full_data["xq"], sorted_full_data["mean"], c='darkblue', alpha=0.5, label='Beta posterior mean')
            two_sigma = 2*np.sqrt(sorted_full_data["vars"])
            kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
            plt.fill_between(sorted_full_data["xq"], (sorted_full_data["mean"]-two_sigma), (sorted_full_data["mean"]+two_sigma), **kwargs)
            plt.locator_params(axis='x', nbins = 6)
            plt.locator_params(axis='y', nbins = 4)
            plt.legend(loc='best')
            plt.title('GP Plot {}_{}'.format(regressors[i], 'full_set'))
            plt.xlabel('Covariate')
            plt.ylabel('Beta Ouput')
            #save plot
            filename = 'GP_{}_{}.pdf'.format(regressors[i], 'full_set')
            file_path = os.path.join(plot_dir, filename)
            plt.savefig(file_path)
        #now save dict entries to a csv_file
        outcsv_name = str(self.epoch).zfill(3) + '_GP_yq_variances.csv'
        covariate_mean_vars_data = pd.DataFrame.from_dict(covariates_mean_vars)
        covariate_mean_vars_data.to_csv(os.path.join(plot_dir, outcsv_name))

    def log_qu_plots(self, log_type):
        """
        Creates q(u) plots which can be passed as figs
        to TB.
        Should be called after each epoch uptade.
        """
        #get means and cov mat S for each covariate
        #x
        qu_m_x = self.gp_params['x']['qu_m'].detach().cpu().numpy().reshape(6)
        qu_S_x = np.diag(self.gp_params['x']['qu_S'].detach().cpu().numpy())
        #y
        qu_m_y = self.gp_params['y']['qu_m'].detach().cpu().numpy().reshape(6)
        qu_S_y = np.diag(self.gp_params['y']['qu_S'].detach().cpu().numpy())
        #z
        qu_m_z = self.gp_params['z']['qu_m'].detach().cpu().numpy().reshape(6)
        qu_S_z = np.diag(self.gp_params['z']['qu_S'].detach().cpu().numpy())
        #xrot
        qu_m_xrot = self.gp_params['xrot']['qu_m'].detach().cpu().numpy().reshape(6)
        qu_S_xrot = np.diag(self.gp_params['xrot']['qu_S'].detach().cpu().numpy())
        #yrot
        qu_m_yrot = self.gp_params['yrot']['qu_m'].detach().cpu().numpy().reshape(6)
        qu_S_yrot = np.diag(self.gp_params['yrot']['qu_S'].detach().cpu().numpy())
        #zrot
        qu_m_zrot = self.gp_params['zrot']['qu_m'].detach().cpu().numpy().reshape(6)
        qu_S_zrot = np.diag(self.gp_params['zrot']['qu_S'].detach().cpu().numpy())

        #now create figure
        fig, axs = plt.subplots(3,2, figsize=(15, 15))

        axs[0,0].plot(self.xu_x.cpu().numpy(), qu_m_x, c='darkblue', alpha=0.5, label = 'q(u) posterior mean')
        x_two_sigma = 2*np.sqrt(qu_S_x)
        kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
        axs[0,0].fill_between(self.xu_x.cpu().numpy(), (qu_m_x-x_two_sigma), (qu_m_x+x_two_sigma), **kwargs)
        axs[0,0].legend(loc='best')
        axs[0,0].set_title('q(u) x covariate at epoch {}'.format(self.epoch))
        axs[0,0].set_xlabel('Covariate x -- x vals ')
        axs[0,0].set_ylabel('q(u)')

        axs[0,1].plot(self.xu_y.cpu().numpy(), qu_m_y, c='darkblue', alpha=0.5, label = 'q(u) posterior mean')
        y_two_sigma = 2*np.sqrt(qu_S_y)
        kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
        axs[0,1].fill_between(self.xu_y.cpu().numpy(), (qu_m_y-y_two_sigma), (qu_m_y+y_two_sigma), **kwargs)
        axs[0,1].legend(loc='best')
        axs[0,1].set_title('q(u) y covariate at epoch {}'.format(self.epoch))
        axs[0,1].set_xlabel('Covariate y -- x vals ')
        axs[0,1].set_ylabel('q(u)')

        axs[1,0].plot(self.xu_z.cpu().numpy(), qu_m_z, c='darkblue', alpha=0.5, label = 'q(u) posterior mean')
        z_two_sigma = 2*np.sqrt(qu_S_z)
        kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
        axs[1,0].fill_between(self.xu_z.cpu().numpy(), (qu_m_z-z_two_sigma), (qu_m_z+z_two_sigma), **kwargs)
        axs[1,0].legend(loc='best')
        axs[1,0].set_title('q(u) z covariate at epoch {}'.format(self.epoch))
        axs[1,0].set_xlabel('Covariate z -- x vals ')
        axs[1,0].set_ylabel('q(u)')

        axs[1,1].plot(self.xu_xrot.cpu().numpy(), qu_m_xrot, c='darkblue', alpha=0.5, label = 'q(u) posterior mean')
        xrot_two_sigma = 2*np.sqrt(qu_S_xrot)
        kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
        axs[1,1].fill_between(self.xu_xrot.cpu().numpy(), (qu_m_xrot-xrot_two_sigma), (qu_m_xrot+xrot_two_sigma), **kwargs)
        axs[1,1].legend(loc='best')
        axs[1,1].set_title('q(u) xrot covariate at epoch {}'.format(self.epoch))
        axs[1,1].set_xlabel('Covariate xrot -- x vals ')
        axs[1,1].set_ylabel('q(u)')

        axs[2,0].plot(self.xu_yrot.cpu().numpy(), qu_m_yrot, c='darkblue', alpha=0.5, label = 'q(u) posterior mean')
        yrot_two_sigma = 2*np.sqrt(qu_S_yrot)
        kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
        axs[2,0].fill_between(self.xu_yrot.cpu().numpy(), (qu_m_yrot-yrot_two_sigma), (qu_m_yrot+yrot_two_sigma), **kwargs)
        axs[2,0].legend(loc='best')
        axs[2,0].set_title('q(u) yrot covariate at epoch {}'.format(self.epoch))
        axs[2,0].set_xlabel('Covariate yrot -- x vals ')
        axs[2,0].set_ylabel('q(u)')

        axs[2,1].plot(self.xu_zrot.cpu().numpy(), qu_m_zrot, c='darkblue', alpha=0.5, label = 'q(u) posterior mean')
        zrot_two_sigma = 2*np.sqrt(qu_S_zrot)
        kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
        axs[2,1].fill_between(self.xu_zrot.cpu().numpy(), (qu_m_zrot-zrot_two_sigma), (qu_m_zrot+zrot_two_sigma), **kwargs)
        axs[2,1].legend(loc='best')
        axs[2,1].set_title('q(u) zrot covariate at epoch {}'.format(self.epoch))
        axs[2,1].set_xlabel('Covariate zrot -- x vals ')
        axs[2,1].set_ylabel('q(u)')

        #and pass it to writer
        self.writer.add_figure("q(u)_{}".format(log_type), fig)

    def log_qkappa_plots(self, log_type):
        """
        Logs q(k) to tensorboard.
        Plots only posterior --> prior is N(1, 0.5).
        Done after each epoch.
        """

        #task
        sa_task = self.gp_params['task']['sa'].detach().cpu().numpy().reshape(1)
        va_task = np.exp(self.gp_params['task']['va'].detach().cpu().numpy())
        task_gauss = norm(sa_task[0], scale = va_task[0])
        x_task = np.linspace(task_gauss.ppf(0.01), task_gauss.ppf(0.99), 100)
        y_task = task_gauss.pdf(x_task)
        #x
        sa_x= self.gp_params['x']['sa'].detach().cpu().numpy().reshape(1)
        va_x = np.exp(self.gp_params['x']['va'].detach().cpu().numpy())
        x_gauss = norm(sa_x[0], scale = va_x[0])
        x_x = np.linspace(x_gauss.ppf(0.01), x_gauss.ppf(0.99), 100)
        y_x = x_gauss.pdf(x_x)
        #y
        sa_y= self.gp_params['y']['sa'].detach().cpu().numpy().reshape(1)
        va_y = np.exp(self.gp_params['y']['va'].detach().cpu().numpy())
        y_gauss = norm(sa_y[0], scale = va_y[0])
        x_y = np.linspace(y_gauss.ppf(0.01), y_gauss.ppf(0.99), 100)
        y_y = y_gauss.pdf(x_y)
        #z
        sa_z= self.gp_params['z']['sa'].detach().cpu().numpy().reshape(1)
        va_z = np.exp(self.gp_params['z']['va'].detach().cpu().numpy())
        z_gauss = norm(sa_z[0], scale = va_z[0])
        x_z = np.linspace(z_gauss.ppf(0.01), z_gauss.ppf(0.99), 100)
        y_z = z_gauss.pdf(x_z)
        #xrot
        sa_xrot= self.gp_params['xrot']['sa'].detach().cpu().numpy().reshape(1)
        va_xrot = np.exp(self.gp_params['xrot']['va'].detach().cpu().numpy())
        xrot_gauss = norm(sa_xrot[0], scale = va_xrot[0])
        x_xrot = np.linspace(xrot_gauss.ppf(0.01), xrot_gauss.ppf(0.99), 100)
        y_xrot = xrot_gauss.pdf(x_xrot)
        #yrot
        sa_yrot= self.gp_params['yrot']['sa'].detach().cpu().numpy().reshape(1)
        va_yrot = np.exp(self.gp_params['yrot']['va'].detach().cpu().numpy())
        yrot_gauss = norm(sa_yrot[0], scale = va_yrot[0])
        x_yrot = np.linspace(yrot_gauss.ppf(0.01), yrot_gauss.ppf(0.99), 100)
        y_yrot = yrot_gauss.pdf(x_yrot)
        #zrot
        sa_zrot= self.gp_params['zrot']['sa'].detach().cpu().numpy().reshape(1)
        va_zrot = np.exp(self.gp_params['zrot']['va'].detach().cpu().numpy())
        zrot_gauss = norm(sa_zrot[0], scale = va_zrot[0])
        x_zrot = np.linspace(zrot_gauss.ppf(0.01), zrot_gauss.ppf(0.99), 100)
        y_zrot = zrot_gauss.pdf(x_zrot)

        #now create plot
        fig, axs = plt.subplots(3,3, figsize=(15, 15))
        axs[0,0].plot(x_task, y_task, lw=2, alpha = 0.5, color = 'green')
        axs[0,0].set_title('Task q(k)')
        axs[0,1].plot(x_x, y_x, lw=2, alpha = 0.5, color = 'blue')
        axs[0,1].set_title('X q(k)')
        axs[0,2].plot(x_y, y_y, lw=2, alpha = 0.5, color = 'orange')
        axs[0,2].set_title('Y q(k)')
        axs[1,0].plot(x_z, y_z, lw=2, alpha = 0.5, color = 'red')
        axs[1,0].set_title('Z q(k)')
        axs[1,1].plot(x_xrot, y_xrot, lw=2, alpha = 0.5, color = 'violet')
        axs[1,1].set_title('Xrot q(k)')
        axs[1,2].plot(x_yrot, y_yrot, lw=2, alpha = 0.5, color = 'magenta')
        axs[1,2].set_title('Yrot q(k)')
        axs[2,0].plot(x_zrot, y_zrot, lw=2, alpha = 0.5, color = 'purple')
        axs[2,0].set_title('Zrot q(k)')
        #pass it to tb writer
        self.writer.add_figure("q(k)_{}".format(log_type), fig)

    def log_beta(self, xq, beta_mean, beta_cov, covariate_name, log_type):
        """
        Logs beta dist plots to TB.
        This is done from within fwd method.
        """
        cov_dict = {}
        xq = xq.cpu().numpy()
        beta_mean = beta_mean.detach().cpu().numpy()
        two_sigma = 2*np.sqrt(np.diag(beta_cov.detach().cpu().numpy()))
        cov_dict['xq'] = xq
        cov_dict['mean'] = beta_mean
        cov_dict['two_sig'] = two_sigma
        cov_data = pd.DataFrame.from_dict(cov_dict)
        sorted_cov_data = cov_data.sort_values(by=["xq"])
        fig = plt.figure()
        plt.plot(sorted_cov_data['xq'], sorted_cov_data['mean'], \
        c='darkblue', alpha=0.5, label='Beta posterior mean')
        kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
        plt.fill_between(sorted_cov_data['xq'], (sorted_cov_data['mean'] - sorted_cov_data['two_sig']), \
        (sorted_cov_data['mean'] + sorted_cov_data['two_sig']), **kwargs)
        plt.legend(loc='best')
        plt.title('Beta_{}'.format(covariate_name))
        plt.xlabel('Covariate')
        plt.ylabel('Beta Ouput')
        self.writer.add_figure("Beta/{}_{}".format(covariate_name, log_type), fig)

    def log_map(self, map, slice, map_name, batch_size, log_type):
        """
        Logs a particular brain map to TB.
        Args
        ----
        Map: np array corresponding to volumetric mk_avg_maps
        for a given minibatch.
        slice: specific slice we wish to log.
        map_name: str.
        Name of map (e.g., base, task...)
        batch_size: int.
        Size of minibatch.
        For now am logging slices only in saggital view.
        """
        map = map.reshape((batch_size, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
        for i in range(batch_size):
            slc = map[i, slice, :, :]
            slc = ndimage.rotate(slc, 90)
            fig_name = '{}_{}_{}/{}'.format(map_name, log_type, slice, i)
            self.writer.add_image(fig_name, slc, dataformats='HW')


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
            self.writer.add_scalar("Loss/Train", loss, self.epoch)
            self.log_qu_plots('train')
            self.log_qkappa_plots('train')
            self.writer.flush()
            # Run through the test data and record a loss.
            if (test_freq is not None) and (epoch % test_freq == 0):
                loss = self.test_epoch(loaders['test'])
                self.loss['test'][epoch] = loss
            # Save the model.
            if (save_freq is not None) and (epoch % save_freq == 0) and (epoch > 0):
                filename = "checkpoint_"+str(epoch).zfill(3)+'.tar'
                file_path = os.path.join(save_dir, filename)
                self.save_state(file_path)
        self.writer.close()

if __name__ == "__main__":
    pass
