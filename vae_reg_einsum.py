"""
Z-based fMRIVAE regression model w/ covariate task as a real variable (i.e, boxcar * HRF)
Added single voxel noise modeling as well using epsilon param
Added motion params in 6 degrees of freedom as regressors of no interest
This should help clean up contrast maps.

March 2020
ToDos
- Try better initilization w/ SPM beta-map ***
- Try regressing out CSF vars as well ***
(once we have dencent cons maps):
- Make it more readable
- Improve documentation
- Implement new save and load state methods
- Add time dependent latent space var plotting
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
import umap
import os
import itertools
from sklearn.decomposition import PCA

IMG_SHAPE = (41,49,35) # maintained shape of original by downsampling data on preprocessing
IMG_DIM = np.prod(IMG_SHAPE)

class VAE(nn.Module):
	def __init__(self, nf=8, save_dir='', lr=1e-4, num_covariates=7, num_latents=32, device_name="auto"):
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
		#init epsilon param for masking
		# -log(10) term accounts for removing model_precision term from original
		epsilon = -np.log(10)*torch.ones([IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]], dtype=torch.float64, device = self.device)
		self.epsilon = torch.nn.Parameter(epsilon)
		#init z_prior
		#When init mean, cov_factor and cov_diag .to(self.device) piece is  NEEDED to vals to be  properly passed to CUDA
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
		#added track_running_stats = False flag to batch_norm _get_layers
		#this improves behavior during test loss calc
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
		#getting z's using encoder
		mu, u, d = self.encode(x)
		latent_dist = LowRankMultivariateNormal(mu, u, d)
		z = latent_dist.rsample()
		base_oh = torch.nn.functional.one_hot(torch.zeros(ids.shape[0], dtype=torch.int64), self.num_covariates+1)
		base_oh = base_oh.to(self.device).float()
		zcat = torch.cat([z, base_oh], 1).float()
		x_rec = self.decode(zcat).view(x.shape[0], -1)
		imgs['base'] = x_rec.detach().cpu().numpy()
		#new_cov = covariates.unsqueeze(-1) #if covar is singleton, use to add batch_dim
		for i in range(1,self.num_covariates+1):
			cov_oh = torch.nn.functional.one_hot(i*torch.ones(ids.shape[0], dtype=torch.int64), self.num_covariates+1)
			cov_oh = cov_oh.to(self.device).float()
			zcat = torch.cat([z, cov_oh], 1).float()
			diff = self.decode(zcat).view(x.shape[0], -1)
			cons = torch.einsum('b,bx->bx', covariates[:, i-1], diff) # using EinSum to preserve batch dim
			x_rec = x_rec + cons
			#now get maps
			keys = list(imgs.keys())
			imgs[keys[i]] = cons.detach().cpu().numpy()
		imgs['full_rec']=x_rec.detach().cpu().numpy()
		# calculating loss
		#new_version, using torch.distributions modules
		elbo = -kl.kl_divergence(latent_dist, self.z_prior) #shape = batch_dim
		#below is long, make it cleaner...
		#obs_dist.shape = batch_dim, img_dim
		obs_dist = Normal(x_rec.float(), torch.exp(-self.epsilon.unsqueeze(0).view(1, -1).expand(ids.shape[0], -1)).float())
		log_prob = obs_dist.log_prob(x.view(ids.shape[0], -1))
		#sum over img_dim to get a batch_dim tensor
		sum_log_prob = torch.sum(log_prob, dim=1)
		elbo = elbo + sum_log_prob
		#contract all values using torch.mean()
		elbo = torch.mean(elbo, dim=0)
		if return_latent_rec:
			return -elbo, z.detach().cpu().numpy(), imgs
		return -elbo

		#original hard-coded version
		#leaving here for now while we test the above
		# this  is missing a log-term for vars epsilon...
		#elbo = -0.5 * (torch.sum(torch.pow(z,2)) + self.num_latents* \
		#np.log(2*np.pi)) # B * p(z)
		#x_diff = x.view(x.shape[0], -1) - x_rec
		#x_diff = torch.einsum('bi,i->bi', x_diff.view(x.shape[0],-1), torch.exp(-self.epsilon.view(-1).float()))
		#elbo = elbo + -0.5 * (torch.sum(torch.pow(x_diff, 2)) + IMG_DIM * \
		#np.log(2*np.pi)) # ~ B * E_{q} p(x|z)
		#elbo = elbo + torch.sum(latent_dist.entropy()) # ~ B * H[q(z|x)]
		#if return_latent_rec:
		#	return -elbo, z.detach().cpu().numpy(), imgs
		#return -elbo

	def train_epoch(self, train_loader):
		self.train()
		train_loss = 0.0
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

#Check out newer save and load methods Jack mentioned
#these will likely be useful here
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

	def project_latent(self, loaders_dict, save_dir, title=None, split=98):
		# plotting only test set since this is un-shuffled. Will plot by subjid in future to overcome this limitation
		#Collect latent means.
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
		c_list = ['b','g','r','c','m','y','k','orange','blueviolet','hotpink','lime','skyblue','teal','sienna']
		colors = itertools.cycle(c_list)
		data_chunks = range(0,len(loaders_dict['test'].dataset),split)  #check value of split here
		#print(data_chunks)
		for i in data_chunks:
			t = np.arange(split)
			plt.scatter(projection[i:i+split,0], projection[i:i+split,1], color=next(colors), s=1.0, alpha=0.6)
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
	def compute_PCA(self, loaders_dict, save_dir):
		csv_file = 'PCA_2n.csv'
		csv_path = os.path.join(save_dir, csv_file)
		latents = np.zeros((len(loaders_dict['test'].dataset), self.num_latents))
		with torch.no_grad():
			j = 0
			for i, sample in enumerate(loaders_dict['test']):
				x = sample['volume']
				x = x.to(self.device)
				mu, _, _ = self.encode(x)
				latents[j:j+len(mu)] = mu.detach().cpu().numpy()
				j += len(mu)
		print("="*40)
		print('Computing latent means PCA')
		# not setting number of components... This should keep all PCs. Max is 32 here
		pca = PCA()
		components = pca.fit_transform(latents)
		print("Number of components: {}".format(pca.n_components_))
		print("Explained variance: {}".format(pca.explained_variance_))

	def reconstruct(self, item, ref_nii, save_dir):
		"""Reconstruct a volume and its cons given a dset idx."""
		x = item['volume'].unsqueeze(0)
		x = x.to(self.device)
		covariates = item['covariates'].unsqueeze(0)
		covariates = covariates.to(self.device)
		ids = item['subjid'].view(1)
		ids = ids.to(self.device)
		with torch.no_grad():
			_, _, imgs = self.forward(ids, covariates, x, return_latent_rec = True) #mk fwrd return base and maps
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
			#Uncomment if adding  PCA calc for each training epoch.
			#self.compute_PCA(loaders_dict=loaders, save_dir = save_dir)
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
