"""
VAE-GAM model implementation.

Gaussian Procress regression implementation is contained separately in the gp.py module.

To train model, plot GPs or create brain maps reconstructions --> use multsubj_reg_run.py as detailed in README.
"""

import gp
import utils
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import nibabel as nib
import numpy as np
import os
import datetime
import pandas as pd
from scipy.stats import norm
from scipy import ndimage
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
from torch.utils.tensorboard import SummaryWriter

IMG_SHAPE = (41,49,35)
IMG_DIM = np.prod(IMG_SHAPE)

class VAE(nn.Module):
    def __init__(self, nf=8, save_dir='', lr=1e-3, num_covariates=7, num_latents=32, device_name="auto", \
    num_inducing_pts=6, gp_kl_scale=10.0, glm_maps = '', glm_reg_scale=1.0, csv_file=''):
        super(VAE, self).__init__()
        self.nf = nf
        self.save_dir = save_dir
        self.lr = lr
        self.num_covariates = num_covariates
        self.num_latents = num_latents
        self.z_dim = self.num_latents + self.num_covariates + 1
        assert device_name != "cuda" or torch.cuda.is_available()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device_name)
        if self.save_dir != '' and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
		# init epsilon variance map.
		# -log(10) initial value accounts for removing model_precision term from Jack's original code.
        epsilon = -np.log(10)*torch.ones([IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]], \
        dtype=torch.float64, device = self.device)
        self.epsilon = torch.nn.Parameter(epsilon)
        #read in GLM maps for regularization term
        glm_maps = pd.read_csv(glm_maps).to_numpy()
        self.glm_maps = torch.from_numpy(glm_maps).to(self.device) #shape is 70,315x8
        self.glm_reg_scale = glm_reg_scale
        self.inducing_pts = num_inducing_pts
        self.gp_kl_scale = torch.as_tensor((gp_kl_scale)).to(self.device)
        # set max lengthscale for 1D GPs.
        self.max_ls = torch.as_tensor(3.0).to(self.device)
        #get ranges for 1D GPs
        xu_ranges = utils.get_xu_ranges(csv_file)
		#init params for GPs
        self.gp_params  = {'task':{}, 'x':{}, 'y':{}, 'z':{}, 'xrot':{}, 'yrot':{}, 'zrot':{}}
        #for task
        #initialize params representing linear term posterior mean (sa) and log std (logstd)
        #no params for non-linear GP piece here --> this is a binary variable!!
        self.sa_task = torch.nn.Parameter(torch.normal(1, 1, size=(1,1)).to(self.device))
        self.gp_params['task']['sa'] = self.sa_task
        self.logstd_task = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['task']['logstd'] = self.logstd_task
		#Now init linear + non-linear (GP) params for remaining (non-binary) regressors.
		#x trans
        self.xu_x = torch.linspace(xu_ranges[0][0], xu_ranges[0][1], self.inducing_pts).to(self.device)
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
        self.logstd_x = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['x']['logstd'] = self.logstd_x
        #y trans
        self.xu_y = torch.linspace(xu_ranges[1][0], xu_ranges[1][1], self.inducing_pts).to(self.device)
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
        self.logstd_y = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['y']['logstd'] = self.logstd_y
        #z trans
        self.xu_z = torch.linspace(xu_ranges[2][0], xu_ranges[2][1], self.inducing_pts).to(self.device)
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
        self.logstd_z = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['z']['logstd'] = self.logstd_z
        #rotational ones
        #xrot
        self.xu_xrot = torch.linspace(xu_ranges[3][0], xu_ranges[3][1], self.inducing_pts).to(self.device)
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
        self.logstd_xrot = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['xrot']['logstd'] = self.logstd_xrot
        #yrot
        self.xu_yrot = torch.linspace(xu_ranges[4][0], xu_ranges[4][1], self.inducing_pts).to(self.device)
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
        self.logstd_yrot = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['yrot']['logstd'] = self.logstd_yrot
        #zrot
        self.xu_zrot = torch.linspace(xu_ranges[5][0], xu_ranges[5][1], self.inducing_pts).to(self.device)
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
        self.logstd_zrot = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['zrot']['logstd'] = self.logstd_zrot
        # init z_prior --> for VAE latents
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

    def calc_linW_KL(self, sa, std):
        """
        Computes KL for linear weight term (\kappa_\alpha).
        This term is added to KL stemming from GP itself, to yield a total GP KL
        contribution to the VAE-GAM objective.
        Outputs:
        KL between prior (N(1, 0.5^2)) and posterior (N(sa, std^2))
        Args
        ----
        sa: posterior mean for a given \kappa_\alpha
        std: posterior standard dev. for a given \kappa_\alpha
        """
        post_dist = Normal(sa, std)
        prior_dist = Normal(1, 0.5)
        qk_kl = kl.kl_divergence(post_dist, prior_dist)
        return qk_kl

    def do_hrf_conv(self, covariate_vals):
        """
        Performs HRF convolution step for biological covariates.
        ------
        Args:
        covariate_vals: torch.tensor
        This is series of 0's or 1's corresponding to covariate being convolved.
        """
        #doing HRF at TR res..
        hrf_times = np.arange(0, 20, 1.4)
        hrf_signal = torch.tensor(utils.hrf(hrf_times)).to(self.device)
        #now create conv matrix --> this is a Toeplitz
        n_time_pts = covariate_vals.shape[0]
        n_hrf_times = hrf_times.shape[0]
        shifted_hrfs = torch.zeros((n_time_pts, (n_time_pts+n_hrf_times-1))).to(self.device)
        for i in range(n_time_pts):
            shifted_hrfs[i, i : i + n_hrf_times] = hrf_signal
        #carry out convolution
        convolved_signal = torch.mm(covariate_vals.unsqueeze(0), shifted_hrfs)
        #take out extra vals and return tensor of appropriate shape/size
        len_to_remove = len(hrf_times) - 1
        convolved_signal = convolved_signal.squeeze(0)[:-len_to_remove]
        return convolved_signal

    def forward(self, ids, covariates, x, log_type, return_latent_rec=False, train_mode=True):
        imgs = {'base': {}, 'task': {}, 'x_mot':{}, 'y_mot':{},'z_mot':{}, 'pitch_mot':{},\
        'roll_mot':{}, 'yaw_mot':{},'full_rec': {}}
        imgs_keys = list(imgs.keys())
        gp_params_keys = list(self.gp_params.keys())
        #set batch gp_kl_loss to zero
        gp_kl_loss = 0
        #set glm regularizer term to zero
        glm_reg = 0
        #getting z's using encoder
        mu, u, d = self.encode(x)
        #check if d is not too small
        #if d is too small, add a small # before using it
        #this solves some numerical instability issues I had at the beginning.
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
        #log base map to TB
        if train_mode:
            #log slices 12, 15 and 18 for all batch elements
            self.log_map(imgs['base'], 12, 'base_map', ids.shape[0], log_type)
            self.log_map(imgs['base'], 15, 'base_map', ids.shape[0], log_type)
            self.log_map(imgs['base'], 18, 'base_map', ids.shape[0], log_type)
        for i in range(1, (self.num_covariates+1)):
            cov_oh = torch.nn.functional.one_hot(i*torch.ones(ids.shape[0],\
            dtype=torch.int64), self.num_covariates+1)
            cov_oh = cov_oh.to(self.device).float()
            zcat = torch.cat([z, cov_oh], 1).float()
            diff = self.decode(zcat).view(x.shape[0], -1)
            #get xqs, these are query pt inputs for our beta (gain) function.
            xq = covariates[:, i-1]
            gp_linW_kl = self.calc_linW_KL(self.gp_params[gp_params_keys[i-1]]['sa'][0], \
            self.gp_params[gp_params_keys[i-1]]['logstd'][0].exp())
            gp_kl_loss += gp_linW_kl
            beta_mean = self.gp_params[gp_params_keys[i-1]]['sa'][0] * xq
            beta_cov = torch.pow(self.gp_params[gp_params_keys[i-1]]['logstd'][0].exp(), 2)* torch.pow(xq, 2) * torch.eye(ids.shape[0]).to(self.device)
            if i!=1:
                #get params for non-linear (GP) piece of gain
                Xu = self.gp_params[gp_params_keys[i-1]]['xu']
                kvar = (self.gp_params[gp_params_keys[i-1]]['logkvar']).exp() + 0.1
                sig = nn.Sigmoid()
                ls = self.max_ls * sig((self.gp_params[gp_params_keys[i-1]]['log_ls']).exp() + 0.5)
                qu_m = self.gp_params[gp_params_keys[i-1]]['qu_m']
                qu_S = self.gp_params[gp_params_keys[i-1]]['qu_S']
                gp_regressor = gp.GP(Xu, kvar, ls, qu_m, qu_S)
                #update location, scale for beta distribution.
                f_bar, Sigma = gp_regressor.evaluate_posterior(xq)
                beta_mean += f_bar
                beta_cov += Sigma
                #now get Kl for non-linear GP term.
                gp_kl = gp_regressor.compute_GP_kl(self.inducing_pts, i, xq, self.save_dir) #adding i, xq and save_dir here to troubleshoot issues with qu_S.
                gp_kl_loss += gp_kl
            beta_dist = MultivariateNormal(beta_mean, (beta_cov + 1e-5*torch.eye(ids.shape[0]).to(self.device))) #added here a small fudge factor for stability.
            task_var = beta_dist.rsample()
            if train_mode:
                #add beta plot to TB
                self.log_beta(xq, beta_mean, beta_cov, gp_params_keys[i-1], log_type)
            #apply HRF conv to biological regressor ONLY
            if i ==1:
                task_var = self.do_hrf_conv(task_var)
            #use result to scale effect map.
            cons = torch.einsum('b,bx->bx', task_var, diff)
            if i==1 and train_mode==True:
                #log only task (vis. stim map) to TB.
                #again, am doing so for slices 12, 15 and 18.
                self.log_map(cons.detach().cpu().numpy(), 12, 'task_map', ids.shape[0], log_type)
                self.log_map(cons.detach().cpu().numpy(), 15, 'task_map', ids.shape[0], log_type)
                self.log_map(cons.detach().cpu().numpy(), 18, 'task_map', ids.shape[0], log_type)
            #encourage all maps to be close to their GLM approximations.
            glm_diff = torch.sum(torch.cdist(cons, self.glm_maps[:, i].unsqueeze(0).expand(ids.shape[0], -1).float(), p=2))
            glm_reg += glm_diff
            x_rec = x_rec + cons
            imgs[imgs_keys[i]] = cons.detach().cpu().numpy()
        imgs['full_rec']=x_rec.detach().cpu().numpy()
        #log full reconstruction to TB as well.
        #again, doing so for slices 12, 15 and 18.
        if train_mode:
            self.log_map(imgs['full_rec'], 12, 'full_reconstruction', ids.shape[0], log_type)
            self.log_map(imgs['full_rec'], 15, 'full_reconstruction', ids.shape[0], log_type)
            self.log_map(imgs['full_rec'], 18, 'full_reconstruction', ids.shape[0], log_type)
        # calculate loss for VAE
        elbo = -kl.kl_divergence(latent_dist, self.z_prior)
        obs_dist = Normal(x_rec.float(),\
        torch.exp(-self.epsilon.unsqueeze(0).view(1, -1).expand(ids.shape[0], -1)).float())
        log_prob = obs_dist.log_prob(x.view(ids.shape[0], -1))
        #sum over img_dim to get a batch_dim tensor
        sum_log_prob = torch.sum(log_prob, dim=1)
        elbo = elbo + sum_log_prob
        #contract all values using torch.mean()
        elbo = torch.mean(elbo, dim=0)
        #now add all losses for compound objective.
        tot_loss = -elbo + self.gp_kl_scale*(gp_kl_loss) + self.glm_reg_scale*(glm_reg)
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
        state['glm_reg_scale'] = self.glm_reg_scale
        state['gp_kl_scale'] = self.gp_kl_scale
        state['inducing_pts'] = self.inducing_pts
        #save only GP_params dict to checkpoint
        #this already includes all linear and non-linear terms we want.
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
        self.epsilon = checkpoint['epsilon']
        self.glm_reg_scale = checkpoint['glm_reg_scale']
        self.gp_kl_scale = checkpoint['gp_kl_scale']
        self.inducing_pts = checkpoint['inducing_pts']
        #load in gp_params dict
        #might need additional step to make each model attr. equal to value
        #in loaded gp_params dict.
        #Double check all vals are readed properly this way...
        self.gp_params = checkpoint['gp_params']

    def project_latent(self, loaders_dict, save_dir, title=None, split=98):
        #Mk sure data is unshuffled when plotting this out!
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
        # UMAP these
        transform = UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
        metric='euclidean', random_state=42)
        projection = transform.fit_transform(latent)
        # And plot them
        c_list = ['b','g','r','c','m','y','k','orange','blueviolet','hotpink',\
        'lime','skyblue','teal','sienna']
        colors = itertools.cycle(c_list)
        data_chunks = range(0,len(loaders_dict['test'].dataset),split)
        for i in data_chunks:
            #commented code below can be used if desiring to plot LS by time pt OR by task/no-task
            #t = np.arange(split)
            #task = np.concatenate((np.zeros(14), np.ones(14)))
            #task = np.tile(task, 3)
            #task = np.concatenate((task, np.zeros(14)))
            #curr version plots LS by subj.
            plt.scatter(projection[i:i+split,0], projection[i:i+split,1],\
            color=next(colors), s=1.0, alpha=0.6)
            #if plotting by task/no-task
            #plt.scatter(projection[i:i+split,0], projection[i:i+split,1],\
            #c=task, s=1.0, alpha=0.6)
            #if plotting by time.
            #plt.scatter(projection[i:i+split,0], projection[i:i+split,1], c=t, s=1.0, alpha=0.6)
            plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.savefig(file_path)
        #return latent, projection

    def reconstruct(self, loader, ref_niis, save_dirs):
        """
        Reconstructs a batch of volumes.
        ------
        Args:
        loader: torch.DataLoader.
        ref_niis: list with paths to reference nifti files for each subj.
        These are used to transform numpy arrays into final nifti format maps.
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
        Plot Posterior mean +/- 2tds for a Beta (Gain) Functions for ALL non-binary covariates.
        Also outputs :
        1) Several csv files (one per non-binary covariate) with SORTED xqs values
        and their corresponding posterior Beta (Gain) mean and variance.

        Parameters
        ---------
        csv_file: file containing data for model. This is the same used for data laoders.
        save_dir: root dir where we wish to save GP plots and csv outputs to.
        """
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
        for i in range(1, len(regressors)):
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
            beta_diag = torch.pow(self.gp_params[regressors[i]]['logstd'][0].exp(), 2)* torch.pow(xq, 2) * torch.eye(xq.shape[0]).to(self.device)
            f_bar, Sigma = gp_regressor.evaluate_posterior(xq)
            beta_mean = (self.gp_params[regressors[i]]['sa'][0] * xq) + f_bar
            beta_cov = beta_diag + Sigma
            #add vals to a covar dict
            curr_cov["xq"] = covariates
            curr_cov["mean"] = beta_mean.detach().cpu().numpy().tolist()
            curr_cov["vars"] = torch.diag(beta_cov).detach().cpu().numpy().tolist()
            #sort vals and save info in csv file.
            outfull_name = str(self.epoch).zfill(3) + '_GP_' + regressors[i] + '_full.csv'
            covariate_full_data = pd.DataFrame.from_dict(curr_cov)
            #sort out predictions
            sorted_full_data = covariate_full_data.sort_values(by=["xq"])
            #save them to csv
            sorted_full_data.to_csv(os.path.join(plot_dir, outfull_name))
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

    def log_qu_plots(self, log_type):
        """
        Creates q(u) plots which can be passed as figs to TB.
        Should be called after each epoch uptade.
        """
        #get means (qu_m) and covariance mat (qu_S) for each covariate
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

        #and pass it to TB writer
        self.writer.add_figure("q(u)_{}".format(log_type), fig)

    def log_qkappa_plots(self, log_type):
        """
        Logs q(k) to tensorboard.
        Plots only posterior --> prior is N(1, 0.5^2).
        """
        #task
        sa_task = self.gp_params['task']['sa'].detach().cpu().numpy().reshape(1)
        std_task = np.exp(self.gp_params['task']['logstd'].detach().cpu().numpy())
        task_gauss = norm(sa_task[0], scale = std_task[0])
        x_task = np.linspace(task_gauss.ppf(0.01), task_gauss.ppf(0.99), 100)
        y_task = task_gauss.pdf(x_task)
        #x
        sa_x= self.gp_params['x']['sa'].detach().cpu().numpy().reshape(1)
        std_x = np.exp(self.gp_params['x']['logstd'].detach().cpu().numpy())
        x_gauss = norm(sa_x[0], scale = std_x[0])
        x_x = np.linspace(x_gauss.ppf(0.01), x_gauss.ppf(0.99), 100)
        y_x = x_gauss.pdf(x_x)
        #y
        sa_y= self.gp_params['y']['sa'].detach().cpu().numpy().reshape(1)
        std_y = np.exp(self.gp_params['y']['logstd'].detach().cpu().numpy())
        y_gauss = norm(sa_y[0], scale = std_y[0])
        x_y = np.linspace(y_gauss.ppf(0.01), y_gauss.ppf(0.99), 100)
        y_y = y_gauss.pdf(x_y)
        #z
        sa_z= self.gp_params['z']['sa'].detach().cpu().numpy().reshape(1)
        std_z = np.exp(self.gp_params['z']['logstd'].detach().cpu().numpy())
        z_gauss = norm(sa_z[0], scale = std_z[0])
        x_z = np.linspace(z_gauss.ppf(0.01), z_gauss.ppf(0.99), 100)
        y_z = z_gauss.pdf(x_z)
        #xrot
        sa_xrot= self.gp_params['xrot']['sa'].detach().cpu().numpy().reshape(1)
        std_xrot = np.exp(self.gp_params['xrot']['logstd'].detach().cpu().numpy())
        xrot_gauss = norm(sa_xrot[0], scale = std_xrot[0])
        x_xrot = np.linspace(xrot_gauss.ppf(0.01), xrot_gauss.ppf(0.99), 100)
        y_xrot = xrot_gauss.pdf(x_xrot)
        #yrot
        sa_yrot= self.gp_params['yrot']['sa'].detach().cpu().numpy().reshape(1)
        std_yrot = np.exp(self.gp_params['yrot']['logstd'].detach().cpu().numpy())
        yrot_gauss = norm(sa_yrot[0], scale = std_yrot[0])
        x_yrot = np.linspace(yrot_gauss.ppf(0.01), yrot_gauss.ppf(0.99), 100)
        y_yrot = yrot_gauss.pdf(x_yrot)
        #zrot
        sa_zrot= self.gp_params['zrot']['sa'].detach().cpu().numpy().reshape(1)
        std_zrot = np.exp(self.gp_params['zrot']['logstd'].detach().cpu().numpy())
        zrot_gauss = norm(sa_zrot[0], scale = std_zrot[0])
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
        #pass it to TB writer
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
        Logs a particular brain map reconstruction to TB.

        Args
        ----
        Map: (np array) map reconstructions for a given minibatch.
        slice: (int) specific slice we wish to log.
        map_name: (string) Name of map (e.g., base, task)
        batch_size: (int) Size of minibatch.
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
