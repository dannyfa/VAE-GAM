"""
Wrapper to call in data class, loaders and vae_reg model

To DO's
- clean up: PCA method call (not needed likely), end reconstruction lines ... 

"""
import os, sys
import argparse
import numpy as np
import random
import torch
#from torch.utils.data import Dataset, DataLoader
import time
import DataClass as data #w/ FMRIDataClass, trsfm and loaders
import vae_reg

parser = argparse.ArgumentParser(description='user args for vae_reg model')

parser.add_argument('--csv_file', type=str, metavar='N', default='/home/dfd4/fmri_vae/resampled/preproc_dset.csv', \
help='Full path to csv file with raw dset to used by DataClass and loaders. This is created by the pre_proc script.')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir where model params, latent projection maps and recon files are saved to. Defaults to saving files in current dir.')
parser.add_argument('--ref_nii', type=str, metavar='N', default='', \
help='Full path to reference nii file to be used for reconstructions.') # did not implement default yet here
parser.add_argument('--batch-size', type=int, default=32, metavar='N', \
help='Input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',\
help='Number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S', \
help='Random seed (default: 1)')
parser.add_argument('--save_freq', type=int, default=10, metavar='N', \
help='How many batches to wait before saving training status')
parser.add_argument('--test_freq', type=int, default=2, metavar='N', \
help='How many batches to wait before testing')
parser.add_argument('--split', type=int, metavar='N', default=98, \
help='split # for project latent method. This is # of frames in each dset.')
#adding beta task map init arg
parser.add_argument('--task_init', type=str, metavar='N', default='', \
help='Path to beta map used to init task contrast in model.')

args = parser.parse_args()
torch.manual_seed(args.seed)

#set up saving directory specs
if args.save_dir =='':
	args.save_dir = os.getcwd()
if args.save_dir != '' and not os.path.exists(args.save_dir):
	os.makedirs(args.save_dir)
else:
	pass

if __name__ == "__main__":
	main_start = time.time()
	loaders_dict = data.setup_data_loaders(batch_size=args.batch_size, csv_file = args.csv_file)
	model = vae_reg.VAE(task_init = args.task_init)
	#uncomment if starting from pre-trained model state
	#model.load_state(filename ='/home/dfd4/fmri_vae/resampled/vaereg_NormRegNoPCA/checkpoint_012.tar')
	model.train_loop(loaders_dict, epochs = args.epochs, test_freq = args.test_freq, save_freq = args.save_freq, save_dir=args.save_dir)
	model.project_latent(loaders_dict, title = "Latent space plot", split=args.split, save_dir=args.save_dir)
	#Uncomment if wanting to do PCA computation after training only (vs. per epoch)
	#model.compute_PCA(loaders_dict, save_dir=args.save_dir)
	data = data.FMRIDataset(csv_file = args.csv_file, transform = data.ToTensor())
	idx= 18 # making this 18th item in dset for now. Will be user input later. This is a vol with task == 1
	item = data.__getitem__(idx)
	model.reconstruct(item, ref_nii=args.ref_nii, save_dir=args.save_dir)
	main_end = time.time()
	print('Total model runtime (seconds): {}'.format(main_end - main_start))
