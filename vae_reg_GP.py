"""
VAE-GAM model module implementation.

For more info on model please see our paper:
 https://www.biorxiv.org/content/10.1101/2021.04.04.438365v2.abstract

Gaussian Procress regression implementation is contained separately in the gp.py module.

To train model, plot GPs or create brain maps use multsubj_reg_run.py as detailed in README.
"""

import gp
import utils
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import nibabel as nib
import numpy as np
import os
import pandas as pd
import itertools
from umap import UMAP
import torch
import torch.utils.data
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal, Normal, kl

IMG_SHAPE = (41,49,35)
IMG_DIM = np.prod(IMG_SHAPE)

class VAE(nn.Module):
    def __init__(self, nf=8, save_dir='', lr=1e-3, num_covariates=7, num_latents=32, device_name="auto", task_init = '', \
    num_inducing_pts=6, mll_scale=10.0, l1_scale=1.0, csv_file=''):
        super(VAE, self).__init__()
        self.nf = nf
        self.save_dir = save_dir
        self.lr = lr
        self.num_covariates = num_covariates
        self.num_latents = num_latents
        self.z_dim = self.num_latents + self.num_covariates + 1
        #adding l1_scale for map regularization term
        self.l1_scale = l1_scale
        assert device_name != "cuda" or torch.cuda.is_available()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device_name)
        if self.save_dir != '' and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
		# init epsilon param modeling single voxel variance
		# -log(10) initial value accounts for removing model_precision term from Jack's original code
        epsilon = -np.log(10)*torch.ones([IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]], \
        dtype=torch.float64, device = self.device)
        self.epsilon = torch.nn.Parameter(epsilon)
		# init. task map as a nn param
		# am using scld avg of 2 task effect maps from SPM as init here.
        # this will be removed from future iterations of this code.
        self.beta_init = torch.FloatTensor(np.array(nib.load(task_init).dataobj)).to(self.device)
        self.task_init = torch.nn.Parameter(self.beta_init)
		#init params for GPs
		#these are Xus (not trainable), Yu's, lengthscale and kernel vars (trainable)
        self.inducing_pts = num_inducing_pts
        self.mll_scale = torch.as_tensor((mll_scale)).to(self.device)
        #Capping ls @10 worked ok for original model. We might wish to tune this down further
        #As it might help avoid our posterior GP cov from failing psd.
        self.max_ls = torch.as_tensor(10.0).to(self.device)
        #construct gp_params dict and init variable vals
        self.gp_params  = utils.build_gp_params_dict(self.inducing_pts, self.device, csv_file)
        # init prior over z's
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

    def gen_zcat(self, batch_dim, map_idx, num_covariates, z):
        """
        Takes in a z tensor sampled from our encoder's posterior and
        concatenates it to a one-hot vector representing either the base map
        or one of the covariates.
        This concatenated vector is fed to the decoder to yield separable maps:
        base + one for each unique covariate.
        Args:
        ---------
        batch_dim: number of entries in a mini-batch.
        map_idx: index corresponding to each map one_hot to be constructed.
        Used 0 for base and 1-7 for task + 6 motion params.
        num_covariates: number of covariates being modelled. This includes nuisance
        covariates such as motion params.
        z: sample of posterior distribution parameterized by encoder.
        """
        one_hot = torch.nn.functional.one_hot(map_idx*torch.ones(batch_dim, dtype=torch.int64),\
        num_covariates).to(self.device).float()
        zcat = torch.cat([z, one_hot], 1).float()
        return zcat

    def calc_ELBO(self, latent_dist, xrec, x, batch_dim):
        """
        Computes VAE ELBO, which will then be added to GP and l1_reg loss terms
        to yield composite objective.
        Args:
        ----
        latent_dist: posterior distribution over latents. Parameterized by encoder.
        x_rec: full reconstruction map output by fwd for a given batch.
        """
        elbo = -kl.kl_divergence(latent_dist, self.z_prior)
        obs_dist = Normal(xrec.float(),\
        torch.exp(-self.epsilon.unsqueeze(0).view(1, -1).expand(batch_dim, -1)).float())
        log_prob = obs_dist.log_prob(x.view(batch_dim, -1))
        #sum over img_dim to get a batch_dim tensor
        sum_log_prob = torch.sum(log_prob, dim=1)
        elbo = elbo + sum_log_prob
        #contract all values using torch.mean()
        elbo = torch.mean(elbo, dim=0)
        return elbo

    def get_gp_outputs(self, idx, xq):
        """
        Gets mean plug in estimator and mll loss for
        a given covariate's GP.
        Args:
        --------
        idx: Float.
        Index for covariate we are currently working with.
        This is used to pass out correct parameters when creating GP object.

        xq: Tensor.
        Input query point values for GP.
        """
        gp_params_keys = list(self.gp_params.keys())
        gp_regressor = gp.GP(self.gp_params[gp_params_keys[idx]]['xu'],\
        self.gp_params[gp_params_keys[idx]]['y'], \
        self.gp_params[gp_params_keys[idx]]['logkvar'],\
        self.gp_params[gp_params_keys[idx]]['log_ls'], self.max_ls)
        #get plug in mean estimator
        y_q, y_var = gp_regressor.predict(xq)
        #get GP mll term
        gp_mll = gp_regressor.calc_mll(self.gp_params[gp_params_keys[idx]]['y'])
        return y_q, y_var, gp_mll


    def forward(self, ids, covariates, x, return_latent_rec=False):
        imgs = {'base': {}, 'task': {}, 'x_mot':{}, 'y_mot':{},'z_mot':{}, 'pitch_mot':{},\
        'roll_mot':{}, 'yaw_mot':{},'full_rec': {}}
        imgs_keys, gp_params_keys = list(imgs.keys()), list(self.gp_params.keys())
        #init batch GP_loss and l1_reg terms to zero
        gp_loss, l1_reg = 0, 0
        #getting z's using encoder
        mu, u, d = self.encode(x)
        #if cov diagtonal output "d" is too small, add some fudge factor to it.
        #this avoids us from getting nan losses when model is ran for a LOT of epochs.
        if len(d[d<1e-6])>= 1:
            d = d.add(1e-6)
        latent_dist = LowRankMultivariateNormal(mu, u, d)
        z = latent_dist.rsample()
        zcat = self.gen_zcat(ids.shape[0], 0, (self.num_covariates+1), z)
        x_rec = self.decode(zcat).view(x.shape[0], -1)
        imgs['base'] = x_rec.detach().cpu().numpy()
        for i in range(1,self.num_covariates+1):
            zcat = self.gen_zcat(ids.shape[0], i, (self.num_covariates+1), z)
            diff = self.decode(zcat).view(x.shape[0], -1)
            #get GP query pts
            xq = covariates[:, i-1]
            #init GP output to zero.
            #this will remain zero for bin variables -- e.g., task
            #but will be altered for continuous ones.
            y_q = torch.zeros(ids.shape[0]).to(self.device)
            #Get GP predictions and GP_mll loss for continuous variables
            if i!=1:
                y_q, _, gp_mll = self.get_gp_outputs((i-1), xq)
                gp_loss += gp_mll
            #scale covariate by linear weigh and add GP prediction as a non-linearity.
            task_var = (self.gp_params[gp_params_keys[i-1]]['linW'] * xq) + y_q
            #convolve this result with HRF
            #this is done for biological regressors only!
            if i ==1:
                task_var = torch.FloatTensor(utils.hrf_convolve(task_var)).to(self.device)
            #now use this result to scale effect map
            cons = torch.einsum('b,bx->bx', task_var, diff)
            #for task covariate, use GLM initialization.
            #this initialization will be eliminated entirely in future versions of this code!
            if i==1:
                cons = cons + self.task_init.unsqueeze(0).contiguous().view(1, -1).expand(ids.shape[0], -1)
            l1_loss = torch.norm(cons, p=1)
            l1_reg += l1_loss
            x_rec = x_rec + cons
            imgs[imgs_keys[i]] = cons.detach().cpu().numpy()
        imgs['full_rec']=x_rec.detach().cpu().numpy()
        #compute VAE ELBO
        elbo = self.calc_ELBO(latent_dist, x_rec, x, ids.shape[0])
        #now generate composite (GP + VAE) loss
        tot_loss = -elbo + self.mll_scale*(-gp_loss) + self.l1_scale*(l1_reg)
        if return_latent_rec:
            return tot_loss, z.detach().cpu().numpy(), imgs
        return tot_loss

    def train_epoch(self, train_loader):
        self.train()
        train_loss = 0.0
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
        state['epsilon'] = self.epsilon
        state['task_init'] = self.task_init
        state['l1_scale'] = self.l1_scale
        state['mll_scale'] = self.mll_scale
        state['inducing_pts'] = self.inducing_pts
        state['gp_params'] = self.gp_params
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
        self.lr = checkpoint['lr']
        self.epsilon = checkpoint['epsilon']
        self.task_init = checkpoint['task_init']
        self.l1_scale = checkpoint['l1_scale']
        self.mll_scale = checkpoint['mll_scale']
        self.inducing_pts = checkpoint['inducing_pts']
        self.gp_params = checkpoint['gp_params']

    def project_latent(self, loaders_dict, save_dir, title=None, split=98):
        filename = str(self.epoch).zfill(3) + '_temp.pdf'
        file_path = os.path.join(save_dir, filename)
        #collect latent means
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
        # Plot.
        c_list = ['b','g','r','c','m','y','k','orange','blueviolet','hotpink',\
        'lime','skyblue','teal','sienna']
        colors = itertools.cycle(c_list)
        data_chunks = range(0,len(loaders_dict['test'].dataset),split)
        for i in data_chunks:
            t = np.arange(split)
            plt.scatter(projection[i:i+split,0], projection[i:i+split,1],\
            color=next(colors), s=1.0, alpha=0.6)
            #commenting plot by time for now...
            #plt.scatter(projection[i:i+split,0], projection[i:i+split,1], c=t, s=1.0, alpha=0.6)
            plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.savefig(file_path)
        #Uncomment line below if we actually wish to get latents and projections
        #return latent, projection

    def reconstruct(self, item, ref_nii, save_dir):
        """
        Reconstruct a volume and its corresponding maps
        for a given  dataset input.
        """
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
        Plot inducing points & posterior mean +/- 2stds for trained GPs.
        Also outputs :
        1) a file containing per covariate GP mean variance
        This info is used by post-processing scripts to merge maps
        of 'flat GP' covariates with base map.
        2) Several csv files (one per covariate) with sorted xqs and
        their corresponding predicted means and variances.

        Args
        ---------
        csv_file: file containing data for model
        this is the same used for data loaders

        """
        #create dict to hold variances for predicted means
        #this is used to decide which covariates have flat GPs and should
        #be incorporated into base map in post-processing.
        keys = ['x_mot', 'y_mot', 'z_mot', 'pitch_mot', 'roll_mot', 'yaw_mot']
        covariates_mean_vars = dict.fromkeys(keys)
        outdir_name = str(self.epoch).zfill(3) + '_GP_plots'
        plot_dir = os.path.join(save_dir, outdir_name)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        data = pd.read_csv(csv_file)
        all_covariates = data[['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z']]
        all_covariates = all_covariates.to_numpy()
        all_covariates = torch.from_numpy(all_covariates)
        regressors = list(self.gp_params.keys())
        for i in range(1, len(regressors)):
            #create dict to hold all entries for each cov -- original input + predicted mean
            #and predicted variances.
            curr_cov = {};
            covariates = all_covariates[:, i-1]
            xq = covariates.to(self.device)
            yq, yvar,_ = self.get_gp_outputs(i, xq)
            #add vals to dict
            curr_cov["xq"] = covariates
            curr_cov["mean"] = yq.detach().cpu().numpy().tolist()
            curr_cov["vars"] = yvar.detach().cpu().numpy().tolist()
            outfull_name = str(self.epoch).zfill(3) + '_GP_' + keys[i-1] + '_full.csv'
            covariate_full_data = pd.DataFrame.from_dict(curr_cov)
            #sort out predictions
            sorted_full_data = covariate_full_data.sort_values(by=["xq"])
            #save them to csv
            sorted_full_data.to_csv(os.path.join(plot_dir, outfull_name))
            #calc variance of predicted GP mean
            #and pass it to dict
            yq_variance = torch.var(yq)
            covariates_mean_vars[keys[i-1]] = [yq_variance.detach().cpu().numpy()]
            x_u = self.gp_params[regressors[i]]['xu'].detach().cpu().numpy()
            y_u = self.gp_params[regressors[i]]['y'].detach().cpu().numpy()
            #create plots and save them
            plt.clf()
            plt.plot(sorted_full_data["xq"], sorted_full_data["mean"], c='darkblue', \
            alpha=0.5, label='posterior mean')
            two_sigma = 2*np.sqrt(sorted_full_data["vars"])
            kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
            plt.fill_between(sorted_full_data["xq"], (sorted_full_data["mean"]-two_sigma),\
            (sorted_full_data["mean"]+two_sigma), **kwargs)
            plt.scatter(x_u, y_u, c='k', label='inducing points')
            plt.locator_params(axis='x', nbins = 6)
            plt.locator_params(axis='y', nbins = 4)
            plt.legend(loc='best')
            plt.title('GP Plot {}_{}'.format(regressors[i], 'full_set'))
            plt.xlabel('Covariate')
            plt.ylabel('GP Prediction')
            filename = 'GP_{}_{}.pdf'.format(regressors[i], 'full_set')
            file_path = os.path.join(plot_dir, filename)
            plt.savefig(file_path)
        #now save predicted mean variances
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
