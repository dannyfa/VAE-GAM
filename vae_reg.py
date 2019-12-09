"""
Z-less fMRIVAE model incorporating covariates sex, age and task type
Should at some point merge w/ Z's from original fMRI_VAE model

November 2019
"""
#uncomment if using matplotlib 3.0 & python3.5 versions
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
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
from torch.distributions import LowRankMultivariateNormal # needed for Jack's v
import umap
import os
import itertools

IMG_SHAPE = (41,49,35) # maintained shape of original by downsampling data
IMG_DIM = np.prod(IMG_SHAPE)

class VAE(nn.Module):
	def __init__(self, nf=8, save_dir='', lr=1e-3, num_subjects=3, num_covariates=3, model_precision=10.0, device_name="auto"):
		super(VAE, self).__init__()
		self.nf = nf
		self.save_dir = save_dir
		self.lr = lr
		self.num_subjects = num_subjects
		self.num_covariates = num_covariates
		self.z_dim = num_subjects + num_covariates + 1 # add one extra dim for base map
		self.model_precision = model_precision
		assert device_name != "cuda" or torch.cuda.is_available()
		if device_name == "auto":
			device_name = "cuda" if torch.cuda.is_available() else "cpu"
			self.device = torch.device(device_name)
		if self.save_dir != '' and not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		self._build_network()
		self.optimizer = Adam(self.parameters(), lr=self.lr)
		self.epoch = 0
		self.loss = {'train':{}, 'test':{}}
		self.to(self.device)

	def _build_network(self):
		#Commented encoder piece for now. Will need it once we incorporate Z's into model
		# Encoder
		#self.conv1 = nn.Conv3d(1,self.nf,3,1)
		#self.conv2 = nn.Conv3d(self.nf,self.nf,3,2)
		#self.conv3 = nn.Conv3d(self.nf,2*self.nf,3,1)
		#self.conv4 = nn.Conv3d(2*self.nf,2*self.nf,3,2)
		#self.conv5 = nn.Conv3d(2*self.nf,2*self.nf,3,1)
		#self.bn1 = nn.BatchNorm3d(1)
		#self.bn3 = nn.BatchNorm3d(self.nf)
		#self.bn5 = nn.BatchNorm3d(2*self.nf)
		#self.fc1 = nn.Linear(2*self.nf*6*8*4, 200)
		#self.fc2 = nn.Linear(200, 100)
		#self.fc31 = nn.Linear(100, 50)
		#self.fc32 = nn.Linear(100, 50)
		#self.fc33 = nn.Linear(100, 50)
		#self.fc41 = nn.Linear(50, self.z_dim)
		#self.fc42 = nn.Linear(50, self.z_dim)
		#self.fc43 = nn.Linear(50, self.z_dim)

		#Decoder
		# In future, implement more flexibility when choosing nn params
		self.fc5 = nn.Linear(self.z_dim, 50) # z_dim would be n+k+1. Here should be 7 - 3 subjs + 3 covariates + base.
		self.fc6 = nn.Linear(50, 100)
		self.fc7 = nn.Linear(100, 200)
		self.fc8 = nn.Linear(200, 2*self.nf*6*8*5)
		self.convt1 = nn.ConvTranspose3d(2*self.nf,2*self.nf,3,1)
		self.convt2 = nn.ConvTranspose3d(2*self.nf,2*self.nf,3,2, padding=(1,0,1), output_padding=(1,0,1))
		self.convt3 = nn.ConvTranspose3d(2*self.nf,self.nf,3,1)
		self.convt4 = nn.ConvTranspose3d(self.nf,self.nf,(5,3,3),2)
		self.convt5 = nn.ConvTranspose3d(self.nf,1,3,1)
		self.bnt1 = nn.BatchNorm3d(2*self.nf)
		self.bnt3 = nn.BatchNorm3d(2*self.nf)
		self.bnt5 = nn.BatchNorm3d(self.nf)

	def _get_layers(self):
		"""Return a dictionary mapping names to network layers.
		Again, adaptions here were minimal -- enough to match layers defined
		in __build_network.
		"""
		#return {'fc1':self.fc1, 'fc2':self.fc2, 'fc31':self.fc31,
        #        'fc32':self.fc32, 'fc33':self.fc33, 'fc41':self.fc41,
        #        'fc42':self.fc42, 'fc43':self.fc43, 'fc5':self.fc5,
        #        'fc6':self.fc6, 'fc7':self.fc7, 'fc8':self.fc8, 'bn1':self.bn1,
        #        'bn3':self.bn3, 'bn5':self.bn5,'bnt1':self.bnt1, 'bnt3':self.bnt3,
        #        'bnt5':self.bnt5, 'conv1':self.conv1,'conv2':self.conv2,
        #        'conv3':self.conv3, 'conv4':self.conv4,
        #        'conv5':self.conv5,'convt1':self.convt1, 'convt2':self.convt2,
        #        'convt3':self.convt3, 'convt4':self.convt4,
        #        'convt5':self.convt5}

		return {'fc5':self.fc5,'fc6':self.fc6, 'fc7':self.fc7, 'fc8':self.fc8,
		        'bnt1':self.bnt1,'bnt3':self.bnt3,'bnt5':self.bnt5, 'convt1':self.convt1,
				'convt2':self.convt2,'convt3':self.convt3, 'convt4':self.convt4,
                'convt5':self.convt5}

    #Commenting encode method for now
	#def encode(self, x):
		#modf so that outpout is in form mu, u, d
		#will try subst. view for squeeze in some pieces here
		#x = x.view(-1,1,IMG_SHAPE[0],IMG_SHAPE[1],IMG_SHAPE[2]) # better to use unsqueeze?
		#h = F.relu(self.conv1(self.bn1(x)))
		#h = F.relu(self.conv2(h))
		#h = F.relu(self.conv3(self.bn3(h)))
		#h = F.relu(self.conv4(h))
		#h = F.relu(self.conv5(self.bn5(h)))
		#h = h.view(-1,2*self.nf*6*8*4)
		#h = F.relu(self.fc1(h))
		#h = F.relu(self.fc2(h))
		#mu = F.relu(self.fc31(h))
		#mu = self.fc41(mu)
		#u = F.relu(self.fc32(h))
		#u = self.fc42(u).unsqueeze(-1) # Last dimension is rank of \Sigma = 1.
		#d = F.relu(self.fc33(h))
		#d = torch.exp(self.fc43(d)) # d must be positive.
		#return mu, u, d

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
		id_oh = torch.nn.functional.one_hot(ids, self.num_subjects)
		cov_oh = torch.nn.functional.one_hot(torch.zeros(ids.shape[0], dtype=torch.int64), self.num_covariates+1)
		cov_oh = cov_oh.to(self.device)
		z = torch.cat([id_oh, cov_oh], 1).float()
		x_rec = self.decode(z).view(x.shape[0], -1)
		for i in range(1,self.num_covariates+1):
			cov_oh = torch.nn.functional.one_hot(i*torch.ones(ids.shape[0], dtype=torch.int64), self.num_covariates+1)
			cov_oh = cov_oh.to(self.device)
			z = torch.cat([id_oh, cov_oh], 1).float()
			diff = self.decode(z).view(x.shape[0], -1)
			x_rec = x_rec + torch.matmul(covariates[:,i-1], diff)
		loss = torch.sum(torch.pow(x.view(x.shape[0],-1) - x_rec, 2))
		if return_latent_rec:
			return -loss, x_rec
		return -loss

	def train_epoch(self, train_loader):
		self.train()
		train_loss = 0.0
		for batch_idx, sample in enumerate(train_loader):
			#Inputs now come from dataloader dicts
			#chnage here to have covariates and ids fed separately?
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

# Not using latents yet ...
#	def project_latent(self, loaders_dict, save_dir, title=None, split=2000):
		# plotting only test since this is unshuffled
		# Collect latent means.
		#filename = 'temp.pdf'
		#file_path = os.path.join(save_dir, filename)
		#latent = np.zeros((len(loaders_dict['test'].dataset), self.z_dim))
		#with torch.no_grad():
		#	j = 0
		#	for i, data in enumerate(loaders_dict['test']):
		#		data = data.to(self.device)
		#		mu, _, _ = self.encode(data)
		#		latent[j:j+len(mu)] = mu.detach().cpu().numpy()
		#		j += len(mu)
		# UMAP them.
		#transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
		#metric='euclidean', random_state=42)
		#projection = transform.fit_transform(latent)
		#print(projection.shape)
		# Plot.
		#c_list = ['b','g','r','c','m','y','k','orange','blueviolet','hotpink','lime','skyblue','teal','sienna']
		#colors = itertools.cycle(c_list)
		#data_chunks = range(0,len(loaders_dict['test'].dataset),split)
		#print(data_chunks)
		#for i in data_chunks:
		#	t = np.arange(split)
			# plt.scatter(projection[i:i+2000,0], projection[i:i+2000,1], color=next(colors), s=1.0, alpha=0.6)
		#	plt.scatter(projection[i:i+split,0], projection[i:i+split,1], c=t, s=1.0, alpha=0.6)
		#	plt.axis('off')
		#if title is not None:
		#	plt.title(title)
		#plt.savefig(file_path)
		# uncomment this if we actually wish to get latent and projections
		#return latent, projection

    #needs to reconstruct both sum, base and separate cons for any given volume
	#inout should be instead a vol #?
	#def reconstruct(self, input_volume, ref_nii, save_dir):
	#	"""Reconstruct the given input volume."""
	#	filename = 'reconstructed_data.nii'
	#	file_path = os.path.join(save_dir, filename)
	#	input_volume = torch.tensor(input_volume, dtype=torch.float64).type(torch.FloatTensor)
	#	input_volume = input_volume.to(self.device)
	#	with torch.no_grad():
	#		_, _, reconstructed = self.forward(input_volume, return_latent_rec = True) #mk fwrd return recon
	#		reconstructed = reconstructed.detach().cpu().numpy()
	#		recon_array = reconstructed.reshape(41,49,35)
			#use nibabel to load in header and affine of filename
			# call that when writing recon_nifti
	#		input_nifti = nib.load(ref_nii)
	#		recon_nifti = nib.Nifti1Image(recon_array, input_nifti.affine, input_nifti.header)
	#		nib.save(recon_nifti, file_path) #add flexibility here for saving in a specific dir

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
				file_path = os.path.join(save_dir, filename) # this line was added, see if this is source of saving issue
				self.save_state(file_path)

if __name__ == "__main__":
	pass
