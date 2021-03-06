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
    def __init__(self, nf=8, save_dir='', lr=1e-3, num_covariates=8, num_latents=32, device_name="auto", \
    num_inducing_pts=6, gp_kl_scale=10.0, glm_maps = '', glm_reg_scale=1.0, csv_files='', neural_covariates=True):
        super(VAE, self).__init__()
        self.nf = nf
        self.save_dir = save_dir
        self.lr = lr
        self.num_covariates = num_covariates
        self.num_latents = num_latents
        self.neural_covariates = neural_covariates
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
        self.glm_maps = torch.from_numpy(glm_maps).to(self.device) #shape is 70,315x9
        self.glm_reg_scale = glm_reg_scale
        self.inducing_pts = num_inducing_pts
        self.gp_kl_scale = torch.as_tensor((gp_kl_scale)).to(self.device)
        # set max lengthscale for 1D GPs.
        self.max_ls = torch.as_tensor(3.0).to(self.device)
        #get ranges for 1D GPs
        xu_ranges = utils.get_xu_ranges(csv_files)
		#init params for GPs
        self.gp_params  = {'task':{}, 'x':{}, 'y':{}, 'z':{}, 'xrot':{}, 'yrot':{}, 'zrot':{}, 'sex':{}}
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
        #init GP (linear term only) for binary sex covariate
        self.sa_sex = torch.nn.Parameter(torch.normal(1, 1, size=(1,1)).to(self.device))
        self.gp_params['sex']['sa'] = self.sa_sex
        self.logstd_sex = torch.nn.Parameter(torch.normal(0, 1, size=(1,1)).to(self.device))
        self.gp_params['sex']['logstd'] = self.logstd_sex
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
        'roll_mot':{}, 'yaw_mot':{}, 'sex':{}, 'full_rec': {}}
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
            utils.log_map(self.writer, IMG_SHAPE, imgs['base'], 12, 'base_map', ids.shape[0], log_type)
            utils.log_map(self.writer, IMG_SHAPE, imgs['base'], 15, 'base_map', ids.shape[0], log_type)
            utils.log_map(self.writer, IMG_SHAPE, imgs['base'], 18, 'base_map', ids.shape[0], log_type)
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
            beta_cov = torch.pow(self.gp_params[gp_params_keys[i-1]]['logstd'][0].exp(), 2)* \
            torch.pow(xq, 2) * torch.eye(ids.shape[0]).to(self.device)
            if i>1 and i<8: #exclude task and sex (binary)
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
                utils.log_beta(self.writer, xq, beta_mean, beta_cov, gp_params_keys[i-1], log_type)
            #apply HRF if biological/neural regressors are present.
            #this set up assumes motion covariates come last and all covariates before that are either
            # 1) biological when neural_covariates==True OR
            # 2) synthetic when neural_covariates==False
            if self.neural_covariates==True and i<(self.num_covariates-6):
                task_var = self.do_hrf_conv(task_var)
            #use result to scale effect map.
            cons = torch.einsum('b,bx->bx', task_var, diff)
            if i==1 and train_mode==True:
                #log only task (vis. stim map) to TB.
                #again, am doing so for slices 12, 15 and 18.
                utils.log_map(self.writer, IMG_SHAPE, cons.detach().cpu().numpy(), 12, 'task_map', ids.shape[0], log_type)
                utils.log_map(self.writer, IMG_SHAPE, cons.detach().cpu().numpy(), 15, 'task_map', ids.shape[0], log_type)
                utils.log_map(self.writer, IMG_SHAPE, cons.detach().cpu().numpy(), 18, 'task_map', ids.shape[0], log_type)
            #encourage all maps to be close to their GLM approximations.
            glm_diff = torch.sum(torch.cdist(cons, self.glm_maps[:, i].unsqueeze(0).expand(ids.shape[0], -1).float(), p=2))
            glm_reg += glm_diff
            x_rec = x_rec + cons
            imgs[imgs_keys[i]] = cons.detach().cpu().numpy()
        imgs['full_rec']=x_rec.detach().cpu().numpy()
        #log full reconstruction to TB as well.
        #again, doing so for slices 12, 15 and 18.
        if train_mode:
            utils.log_map(self.writer, IMG_SHAPE, imgs['full_rec'], 12, 'full_reconstruction', ids.shape[0], log_type)
            utils.log_map(self.writer, IMG_SHAPE, imgs['full_rec'], 15, 'full_reconstruction', ids.shape[0], log_type)
            utils.log_map(self.writer, IMG_SHAPE, imgs['full_rec'], 18, 'full_reconstruction', ids.shape[0], log_type)
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
        self.gp_params = checkpoint['gp_params']
        # and individual 1D gp params
        #these are needed for gradient to pass through appropriately
        #when training from existing checkpoint.
        #task
        self.sa_task = checkpoint['gp_params']['task']['sa']
        self.logstd_task = checkpoint['gp_params']['task']['logstd']
        # x_trans
        self.sa_x = checkpoint['gp_params']['x']['sa']
        self.logstd_x = checkpoint['gp_params']['x']['logstd']
        self.qu_m_x = checkpoint['gp_params']['x']['qu_m']
        self.qu_S_x = checkpoint['gp_params']['x']['qu_S']
        self.logkvar_x = checkpoint['gp_params']['x']['logkvar']
        self.logls_x = checkpoint['gp_params']['x']['log_ls']
        #y_trans
        self.sa_y = checkpoint['gp_params']['y']['sa']
        self.logstd_y = checkpoint['gp_params']['y']['logstd']
        self.qu_m_y = checkpoint['gp_params']['y']['qu_m']
        self.qu_S_y = checkpoint['gp_params']['y']['qu_S']
        self.logkvar_y = checkpoint['gp_params']['y']['logkvar']
        self.logls_y = checkpoint['gp_params']['y']['log_ls']
        #z_trans
        self.sa_z = checkpoint['gp_params']['z']['sa']
        self.logstd_z = checkpoint['gp_params']['z']['logstd']
        self.qu_m_z = checkpoint['gp_params']['z']['qu_m']
        self.qu_S_z = checkpoint['gp_params']['z']['qu_S']
        self.logkvar_z = checkpoint['gp_params']['z']['logkvar']
        self.logls_z = checkpoint['gp_params']['z']['log_ls']
        #xrot
        self.sa_xrot= checkpoint['gp_params']['xrot']['sa']
        self.logstd_xrot = checkpoint['gp_params']['xrot']['logstd']
        self.qu_m_xrot = checkpoint['gp_params']['xrot']['qu_m']
        self.qu_S_xrot = checkpoint['gp_params']['xrot']['qu_S']
        self.logkvar_xrot = checkpoint['gp_params']['xrot']['logkvar']
        self.logls_xrot = checkpoint['gp_params']['xrot']['log_ls']
        #yrot
        self.sa_yrot = checkpoint['gp_params']['yrot']['sa']
        self.logstd_yrot = checkpoint['gp_params']['yrot']['logstd']
        self.qu_m_yrot = checkpoint['gp_params']['yrot']['qu_m']
        self.qu_S_yrot = checkpoint['gp_params']['yrot']['qu_S']
        self.logkvar_yrot = checkpoint['gp_params']['yrot']['logkvar']
        self.logls_yrot = checkpoint['gp_params']['yrot']['log_ls']
        #zrot
        self.sa_zrot = checkpoint['gp_params']['zrot']['sa']
        self.logstd_zrot = checkpoint['gp_params']['zrot']['logstd']
        self.qu_m_zrot = checkpoint['gp_params']['zrot']['qu_m']
        self.qu_S_zrot = checkpoint['gp_params']['zrot']['qu_S']
        self.logkvar_zrot = checkpoint['gp_params']['zrot']['logkvar']
        self.logls_zrot = checkpoint['gp_params']['zrot']['log_ls']
        #sex
        self.sa_sex = checkpoint['gp_params']['sex']['sa']
        self.logstd_sex = checkpoint['gp_params']['sex']['logstd']


    def project_latent(self, loaders_dict, save_dir, title=None, split=98):
        #Mk sure data is unshuffled when plotting this out!
        # Collect latent means.
        filename = str(self.epoch).zfill(3) + '_temp.pdf'
        file_path = os.path.join(save_dir, filename)
        latent = np.zeros((len(loaders_dict['UnShuffled_train'].dataset), self.num_latents))
        with torch.no_grad():
            j = 0
            for i, sample in enumerate(loaders_dict['UnShuffled_train']):
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
        data_chunks = range(0,len(loaders_dict['UnShuffled_train'].dataset),split)
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
        for i in range(1, len(regressors)-1): #skip task and sex covariates (binary)
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

    def train_loop(self, loaders, epochs=100, test_freq=2, save_freq=10, save_dir = ''):
        print("="*40)
        print("Training: epochs", self.epoch, "to", self.epoch+epochs-1)
        print("Training set:", len(loaders['Shuffled_train'].dataset))
        print("Test set:", len(loaders['test'].dataset))
        print("="*40)
        # For some number of epochs...
        for epoch in range(self.epoch, self.epoch+epochs):
            #Run through the training data and record a loss.
            loss = self.train_epoch(loaders['Shuffled_train'])
            self.loss['train'][epoch] = loss
            self.writer.add_scalar("Loss/Train", loss, self.epoch)
            utils.log_qu_plots(self.epoch, self.gp_params, self.writer, 'train')
            utils.log_qkappa_plots(self.gp_params, self.writer, 'train')
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
