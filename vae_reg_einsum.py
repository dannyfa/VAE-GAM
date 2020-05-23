"""
Z-based fMRIVAE regression model w/ task as a real variable (i.e, boxcar * HRF)
- Added single voxel noise modeling (epsilon param)
- Added motion regressors in 6 degrees of freedom (from fmriprep)
- Added 1D GPs to model regressors (task + 6 motion params)
- Added initilization using and avg of SPM's task beta map slcd to take only 11% of total explained variance

To Do's
- Make code less redundant
- Improve GP plotting method (do it over entire set, without wasting CPU while looping though dataloaders )
- Add time dependent latent space plotting (post-NIPs)
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
from torch.distributions import LowRankMultivariateNormal, Normal, kl
#uncomment if needing to chase-down a nan in loss
#from torch import autograd
import umap
import os
import itertools
from sklearn.decomposition import PCA
import gp #module with GP class

# maintained shape of original nn by downsampling data on preprocessing
IMG_SHAPE = (41,49,35)
IMG_DIM = np.prod(IMG_SHAPE)

class VAE(nn.Module):
	def __init__(self, nf=8, save_dir='', lr=1e-3, num_covariates=7, num_latents=32, device_name="auto", task_init = '', num_inducing_pts=10.0, mll_scale=2.0):
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
		# init epsilon param modeling single voxel variance
		# -log(10) init. accounts for removing model_precision term from original version
		epsilon = -np.log(10)*torch.ones([IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]], dtype=torch.float64, device = self.device)
		self.epsilon = torch.nn.Parameter(epsilon)
		# init. task cons as a nn param
		# am using variance scld avg of task effect map from SPM as init here.
		beta_init = np.array(nib.load(task_init).dataobj)
		self.task_init = torch.nn.Parameter(torch.FloatTensor(beta_init).to(self.device))
		#init params for GPs
		#these are Xus (not trainable), Yu's, lengthscale and kernel vars (trainable)
		#pass these to a big dict -- gp_params
		#took out obs noise
		#self.y_var = torch.as_tensor((0.1)).to(self.device)
		#testing increasing mll scale
		self.inducing_pts = num_inducing_pts
		self.mll_scale = torch.as_tensor((mll_scale)).to(self.device)
		self.gp_params  = {'task':{}, 'x':{}, 'y':{}, 'z':{}, 'xrot':{}, 'yrot':{}, 'zrot':{}}
		#for task
		#reducing num of inducing points to 10
		self.xu_task = torch.linspace(-0.5, 2.5, self.inducing_pts).to(self.device)
		self.gp_params['task']['xu'] = self.xu_task
		self.Y_task = torch.nn.Parameter(torch.rand(self.inducing_pts).to(self.device))
		self.gp_params['task']['y'] = self.Y_task
		self.kvar_task = torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
		self.gp_params['task']['kvar'] = self.kvar_task
		self.logls_task = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
		self.gp_params['task']['log_ls'] = self.logls_task
		#Now same for 6 motion GPs
		#x trans
		#reducing num of inducing pts to 10
		self.xu_x = torch.linspace(-0.2, 0.2, self.inducing_pts).to(self.device)
		self.gp_params['x']['xu'] = self.xu_x
		self.Y_x = torch.nn.Parameter(torch.rand(self.inducing_pts).to(self.device))
		self.gp_params['x']['y'] = self.Y_x
		self.kvar_x = torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
		self.gp_params['x']['kvar'] = self.kvar_x
		self.logls_x = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
		self.gp_params['x']['log_ls'] = self.logls_x
        #y trans
		self.xu_y = torch.linspace(-0.4, 0.4, self.inducing_pts).to(self.device)
		self.gp_params['y']['xu'] = self.xu_y
		self.Y_y = torch.nn.Parameter(torch.rand(self.inducing_pts).to(self.device))
		self.gp_params['y']['y'] = self.Y_y
		self.kvar_y = torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
		self.gp_params['y']['kvar'] = self.kvar_y
		self.logls_y = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
		self.gp_params['y']['log_ls'] = self.logls_y
		#z trans
		self.xu_z = torch.linspace(-0.5, 0.5, self.inducing_pts).to(self.device)
		self.gp_params['z']['xu'] = self.xu_z
		self.Y_z = torch.nn.Parameter(torch.rand(self.inducing_pts).to(self.device))
		self.gp_params['z']['y'] = self.Y_z
		self.kvar_z = torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
		self.gp_params['z']['kvar'] = self.kvar_z
		self.logls_z = torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
		self.gp_params['z']['log_ls'] = self.logls_z
		#rotational ones
		#xrot
		#reducing number of inducing points to 10
		self.xu_xrot = torch.linspace(-0.01, 0.01, self.inducing_pts).to(self.device)
		self.gp_params['xrot']['xu'] = self.xu_xrot
		self.Y_xrot = torch.nn.Parameter(torch.rand(self.inducing_pts).to(self.device))
		self.gp_params['xrot']['y'] = self.Y_xrot
		self.kvar_xrot= torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
		self.gp_params['xrot']['kvar'] = self.kvar_xrot
		self.logls_xrot= torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
		self.gp_params['xrot']['log_ls'] = self.logls_xrot
		#yrot
		self.xu_yrot = torch.linspace(-0.005, 0.005, self.inducing_pts).to(self.device)
		self.gp_params['yrot']['xu'] = self.xu_yrot
		self.Y_yrot= torch.nn.Parameter(torch.rand(self.inducing_pts).to(self.device))
		self.gp_params['yrot']['y'] = self.Y_yrot
		self.kvar_yrot = torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
		self.gp_params['yrot']['kvar'] = self.kvar_yrot
		self.logls_yrot= torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
		self.gp_params['yrot']['log_ls'] = self.logls_yrot
		#zrot
		self.xu_zrot = torch.linspace(-0.005, 0.005, self.inducing_pts).to(self.device)
		self.gp_params['zrot']['xu'] = self.xu_zrot
		self.Y_zrot= torch.nn.Parameter(torch.rand(self.inducing_pts).to(self.device))
		self.gp_params['zrot']['y'] = self.Y_zrot
		self.kvar_zrot= torch.nn.Parameter(torch.as_tensor((1.0)).to(self.device))
		self.gp_params['zrot']['kvar'] = self.kvar_zrot
		self.logls_zrot= torch.nn.Parameter(torch.as_tensor((0.0)).to(self.device))
		self.gp_params['zrot']['log_ls'] = self.logls_zrot
		# init z_prior
		# When init mean, cov_factor and cov_diag '.to(self.device)' piece is  NEEDED for vals to be  properly passed to CUDA...
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


	def forward(self, ids, covariates, x, return_latent_rec=False):
		imgs = {'base': {}, 'task': {}, 'x_mot':{}, 'y_mot':{},'z_mot':{}, 'pitch_mot':{},\
		'roll_mot':{}, 'yaw_mot':{},'full_rec': {}}
		imgs_keys = list(imgs.keys())
		gp_params_keys = list(self.gp_params.keys())
		#set batch GP_loss to zero
		#computed mlls will be added to it and passed on to overall batch_loss along w/ VAE loss
		gp_loss = 0
		#getting z's using encoder
		mu, u, d = self.encode(x)
		#check if d is not too small
		#if d is too small, add a small # before using it
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
			cov_oh = torch.nn.functional.one_hot(i*torch.ones(ids.shape[0],\
			dtype=torch.int64), self.num_covariates+1)
			cov_oh = cov_oh.to(self.device).float()
			zcat = torch.cat([z, cov_oh], 1).float()
			diff = self.decode(zcat).view(x.shape[0], -1)
			#get params for GP regressor
			Xu = self.gp_params[gp_params_keys[i-1]]['xu']
			Yu = self.gp_params[gp_params_keys[i-1]]['y']
			kvar = self.gp_params[gp_params_keys[i-1]]['kvar']
			#assert kernel ls is at a minimum some small positive number
			#this avoids issues with getting a singular mat during GP cholesky decomposition
			ls = (self.gp_params[gp_params_keys[i-1]]['log_ls']).exp() + 0.5
			#instantiate GP object
			gp_regressor = gp.GP(Xu, Yu, kvar, ls)
			#get xqs, these are inputs for query pts
			#effectively these are just values of covariates for a given batch
			xq = covariates[:, i-1]
			#calc predictions for query points
			y_q, y_vars = gp_regressor.predict(xq)
			#calc marginal likelihood for gp
			gp_mll = gp_regressor.calc_mll(Yu)
			#add it to gp_loss term
			gp_loss += gp_mll
			#add residual prediction from GP to task variable
			task_var = covariates[:, i-1] + y_q
			# use this to scale effect map
			#using EinSum to preserve batch dim
			cons = torch.einsum('b,bx->bx', task_var, diff)
			#add cons to init_task param if covariate == 'task'
			#implementation below was adopted to avoid in place ops that would cause autograd errors
			if i==1:
				cons = cons + self.task_init.unsqueeze(0).view(1, -1).expand(ids.shape[0], -1)
			x_rec = x_rec + cons
			imgs[imgs_keys[i]] = cons.detach().cpu().numpy()
		imgs['full_rec']=x_rec.detach().cpu().numpy()
		# calculating loss for VAE ...
		# This version uses torch.distributions modules only
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
		#scalling factor is a hyperparam
		tot_loss = -elbo + self.mll_scale*(-gp_loss)
		if return_latent_rec:
			return tot_loss, z.detach().cpu().numpy(), imgs
		return tot_loss

    #commented autograd.detect.anomaly() line  was used to trace out nan loss issue
	#only use this if trying to trace issues with auto-grad
	#otherwise, it will significantly slow code execution!

	def train_epoch(self, train_loader):
		self.train()
		train_loss = 0.0
		#with autograd.detect_anomaly():
		for batch_idx, sample in enumerate(train_loader):
			#Inputs now come from dataloader dicts
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
		#add GP nn params to checkpt files
		state['Y_task'] = self.Y_task
		state['kvar_task'] = self.kvar_task
		state['logls_task'] = self.logls_task
		state['Y_x'] = self.Y_x
		state['kvar_x'] = self.kvar_x
		state['logls_x'] = self.logls_x
		state['Y_y'] = self.Y_y
		state['kvar_y'] = self.kvar_y
		state['logls_y'] = self.logls_y
		state['Y_z'] = self.Y_z
		state['kvar_z'] = self.kvar_z
		state['logls_z'] = self.logls_z
		state['Y_xrot'] = self.Y_xrot
		state['kvar_xrot'] = self.kvar_xrot
		state['logls_xrot'] = self.logls_xrot
		state['Y_yrot'] = self.Y_yrot
		state['kvar_yrot'] = self.kvar_yrot
		state['logls_yrot'] = self.logls_yrot
		state['Y_zrot'] = self.Y_zrot
		state['kvar_zrot'] = self.kvar_zrot
		state['logls_zrot'] = self.logls_zrot
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
		self.task_init = checkpoint['task_init']
		#load in GP params from ckpt files
		self.Y_task = checkpoint['Y_task']
		self.kvar_task = checkpoint['kvar_task']
		self.logls_task = checkpoint['logls_task']
		self.Y_x = checkpoint['Y_x']
		self.kvar_x = checkpoint['kvar_x']
		self.logls_x = checkpoint['logls_x']
		self.Y_y = checkpoint['Y_y']
		self.kvar_y = checkpoint['kvar_y']
		self.logls_y = checkpoint['logls_y']
		self.Y_z = checkpoint['Y_z']
		self.kvar_z = checkpoint['kvar_z']
		self.logls_z = checkpoint['logls_z']
		self.Y_xrot = checkpoint['Y_xrot']
		self.kvar_xrot = checkpoint['kvar_xrot']
		self.logls_xrot = checkpoint['logls_xrot']
		self.Y_yrot = checkpoint['Y_yrot']
		self.kvar_yrot = checkpoint['kvar_yrot']
		self.logls_yrot= checkpoint['logls_yrot']
		self.Y_zrot = checkpoint['Y_zrot']
		self.kvar_zrot = checkpoint['kvar_zrot']
		self.logls_zrot = checkpoint['logls_zrot']
		self.mll_scale = checkpoint['mll_scale']
		self.inducing_pts = checkpoint['inducing_pts']

	def project_latent(self, loaders_dict, save_dir, title=None, split=98):
		# plotting only test set since this is un-shuffled.
		# Will plot by subjid in future to overcome this limitation
		# Collect latent means.
		filename = 'temp.pdf'
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
		transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
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
			plt.scatter(projection[i:i+split,0], projection[i:i+split,1], \
			color=next(colors), s=1.0, alpha=0.6)
			#commenting plot by time
			#plt.scatter(projection[i:i+split,0], projection[i:i+split,1], c=t, s=1.0, alpha=0.6)
			plt.axis('off')
		if title is not None:
			plt.title(title)
		plt.savefig(file_path)
		#Uncomment this if we actually wish to get latent and projections
		#return latent, projection

    # Adding new method to compute PCA for latent means.
	# This can be done per epoch or after some training
	#cmt since no longer needed
	#def compute_PCA(self, loaders_dict, save_dir):
	#	csv_file = 'PCA_2n.csv'
	#	csv_path = os.path.join(save_dir, csv_file)
	#	latents = np.zeros((len(loaders_dict['test'].dataset), self.num_latents))
	#	with torch.no_grad():
	#		j = 0
	#		for i, sample in enumerate(loaders_dict['test']):
	#			x = sample['volume']
	#			x = x.to(self.device)
	#			mu, _, _ = self.encode(x)
	#			latents[j:j+len(mu)] = mu.detach().cpu().numpy()
	#			j += len(mu)
	#	print("="*40)
	#	print('Computing latent means PCA')
		# not setting number of components... This should keep all PCs. Max is 32 here
	#	pca = PCA()
	#	components = pca.fit_transform(latents)
	#	print("Number of components: {}".format(pca.n_components_))
	#	print("Explained variance: {}".format(pca.explained_variance_))

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

	def plot_GPs(self, loaders_dict, save_dir = ''):
		"""
		Plot inducing points &
		posterior mean +/- 2tds for a trained GPs

		Parameters
		----------
		loaders_dict: torch dataloader dict
		              Divided up into train and test

		ToDo's: find more efficient way to plot these
				looping though DataLoader fast is NOT a good idea...
		"""
		regressors = list(self.gp_params.keys())
		for i in range(len(regressors)):
			#build GP for the regressor
			xu = self.gp_params[regressors[i]]['xu']
			yu = self.gp_params[regressors[i]]['y']
			kvar = self.gp_params[regressors[i]]['kvar']
			ls = (self.gp_params[regressors[i]]['log_ls']).exp() + 0.5
			gp_regressor = gp.GP(xu, yu, kvar, ls)
			for j, sample in enumerate(loaders_dict['test']):
				if j <= 0:
					covariates = sample['covariates']
					covariates = covariates.to(self.device)
					xq = covariates[:, i]
					yq, yvar = gp_regressor.predict(xq)
					#pass vars to cpu and np prior to plotting
					x_u = xu.detach().cpu().numpy()
					y_u = yu.detach().cpu().numpy()
					x_q = xq.detach().cpu().numpy()
					y_q = yq.detach().cpu().numpy()
					y_var = yvar.detach().cpu().numpy()
					#create plot and save it
					plt.scatter(x_u, y_u, c='k', label='inducing points')
					plt.plot(x_q, y_q, c='b', alpha=0.6, label='posterior mean')
					two_sigma = 2*np.sqrt(y_var)
					kwargs = {'color':'b', 'alpha':0.2, 'label':'2 sigma'}
					plt.fill_between(x_q, y_q-two_sigma, y_q+two_sigma, **kwargs)
					plt.legend(loc='best')
					plt.title('GP Plot {}_{}'.format(regressors[i], str(int(j))))
					plt.xlabel('X')
					plt.ylabel('Y')
					#save plot & clean it ...
					plot_dir = os.path.join(save_dir, 'GP_plots')
					if not os.path.exists(plot_dir):
						os.makedirs(plot_dir)
					filename = 'GP_{}_{}.pdf'.format(regressors[i], str(int(j)))
					file_path = os.path.join(plot_dir, filename)
					plt.savefig(file_path)
					plt.clf()
				else:
					pass

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
			# Uncomment if adding  PCA calc for each training epoch.
			# self.compute_PCA(loaders_dict=loaders, save_dir = save_dir)
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
