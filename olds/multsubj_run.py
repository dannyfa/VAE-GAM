"""
Script for calling running fMRI VAE on multiple subj
Also a modificaiton of Rachel's prior work!

November 2019
"""
import os, sys
import argparse
import numpy as np
import glob
import random
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import time
import VAE_fmri #w/ VAE class

#Get args from user
#Took out --yes(no)-cuda args --> these are implemented inside Jack's code
#Added a couple  of new args:
#1)data_dir --  where .nii files live
#2)save_freq
#3)test_freq
#4)ref_nii
#see short descriptions below

parser = argparse.ArgumentParser(description='user args for fMRI VAE')

parser.add_argument('--data_dir', type=str, metavar='N', default='', \
help='Dir where nifti files to be used are located. Assumes files inside end in ext trimmed*.nii.gz.')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', \
help='Input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',\
help='Number of epochs to train (default: 100)')
parser.add_argument('--seed', type=int, default=1, metavar='S', \
help='Random seed (default: 1)')
parser.add_argument('--save_freq', type=int, default=10, metavar='N', \
help='How many batches to wait before saving training status')
parser.add_argument('--test_freq', type=int, default=2, metavar='N', \
help='How many batches to wait before testing')
#parser.add_argument('--ref_nii', type=str, metavar='N', \
#help='Reference nifti file used to reconstruct dset. If file not on scripts dir, full path must be given') #implement this. Needs default?
parser.add_argument('--split', type=int, metavar='N', default=2000, \
help='split # for train and test dataloaders.  This is # of frames in trimmed dsets.')


args = parser.parse_args()
torch.manual_seed(args.seed)

#get list of .nii files
if args.data_dir!='':
	if os.path.exists(args.data_dir):
		#can mk file ext. a flexible input as well.
		#For now this should do
		try:
			full_path = os.path.join(args.data_dir,'trimmed*.nii.gz')
			file_list=glob.glob(full_path)
			sampled_files = random.sample(file_list, 7) #take 7 files randomly from data_dir
			ref_nii = sampled_files[0]
		except:
			print('Did not find expected trimmed*.nii.gz files on data_dir!')
			print('Cannot proceed w/out data files!')
			sys.exit()
	else:
		print('Ooops. Data directory given does not exist!')
		print('Cannot proceed w/out data files!')
		sys.exit()

else:
	full_path = os.path.join(os.getcwd(),'trimmed*.nii.gz')
	try:
		file_list=glob.glob(full_path)
		sampled_files = random.sample(file_list, 7) #take 7 files randomly from data_dir
		ref_nii = sampled_files[0]
	except:
		print('Did not find expected trimmed*.nii.gz files on cwd!')
		print('Cannot proceed w/out data files!')
		sys.exit()

##Concat all files into np array
print('Loading data...')
print(sampled_files)
#data_list = [nib.load(sampled_files[file_idx]).dataobj for file_idx in range(len(sampled_files))]
print([np.array(nib.load(sampled_files[file_idx]).dataobj for file_idx in range(len(sampled_files)))])
dataset = np.concatenate([np.array(nib.load(sampled_files[file_idx]).dataobj) for file_idx in range(len(sampled_files))], axis=3)
maxsig = 65536 # Need to add function that calc. max needed here. Hardcoded for now!!
dataset = np.true_divide(dataset, maxsig)

#set up FMRIDataset Class and DataLoaders
#Will expand on FMRDataset class to add functionality
#Eventually these will be described in a separate module and  imported ...

class FMRIDataset(Dataset):
	"""TO DO: Real FMRI dataset"""
	def __init__(self, filenames, frames_per_file):
		self.filenames = filenames
		self.frames_per_file = frames_per_file

	def __len__(self):
		"""Return the number of frames."""
		return len(self.filenames) * self.frames_per_file

	def __getitem__(self, index):
		"""Return a single time frame as a torch tensor."""
		file_index = index // self.frames_per_file
		within_file_index = index
		return torch.tensor(dataset[:,:,:,within_file_index], \
        requires_grad=False, dtype=torch.float64).type(torch.FloatTensor)

def setup_data_loaders(batch_size=32, shuffle=(True, False), split=2000):
	# Setup the train loaders.
	train_dataset = FMRIDataset(sampled_files, split)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, \
    shuffle=shuffle[0], num_workers=14) #changed to 1
	# Setup the test loaders.
	test_dataset = FMRIDataset(sampled_files, split)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, \
    shuffle=shuffle[1], num_workers=14)
	#return dict w/ train & test loaders -- helps w/ train_loop method from Jack
	return {'train':train_loader, 'test':test_loader}

if __name__ == "__main__":
	main_start = time.time()
	loaders_dict = setup_data_loaders(batch_size=args.batch_size, split=args.split)
	model = VAE_fmri.VAE()
	model.train_loop(loaders_dict, epochs = args.epochs, test_freq = args.test_freq, save_freq = args.save_freq)
	model.project_latent(loaders_dict,filename = 'nv_temp_test2.pdf', title = "New version test2", split=args.split)
	#line for old version of project_latent
	#model.project_latent(loaders_dict['test'], 'latent_project_plot.pdf') # see if need to use new color version
	#for now reconstructing 100th volume in dset.
	# Eventually user should be able to specifiy volume(s) to be reconstructed
	# ref_nii is currently not a user  input but a sample .nii from data_dir.
	model.reconstruct(dataset[:,:,:,100], ref_nii)
	# Add more refined option here for loading pre-trained model
	# model.load_state('checkpoint.tar')
	main_end = time.time()
	print('Total model runtime (seconds): {}'.format(main_end - main_start))
