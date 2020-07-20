"""
Wrapper to call in data class, loaders and vae_reg model
ToDo:
- merge versions w/ and out initialization option
- modificatiosn needed both here and on vae_reg_GP model
"""
import os, sys
import argparse
import numpy as np
import random
import torch
import time
import DataClass_GP as data #w/ FMRIDataClass, trsfm and loaders
import vae_reg_GP_noinit as vae_reg
import build_model_recons as recon

parser = argparse.ArgumentParser(description='user args for vae_reg model')

parser.add_argument('--csv_file', type=str, metavar='N', default='', \
help='Full path to csv file with raw dset to used by DataClass and loaders. This is created by the pre_proc script.')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir where model params, GP plots, latent projection maps and recon files are saved to. Defaults to saving files in current dir.')
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
parser.add_argument('--split', type=int, metavar='N', default=284, \
help='split # for project latent method. This is # of frames in each dset.')
#adding args for number of inducing points for GPs and mll scale
#these are mostly useful for troubleshooting GP Training
parser.add_argument('--num_inducing_pts', type=int, metavar='N', default=6, \
help='Number of inducing points for regressor GPs.')
parser.add_argument('--mll_scale', type=float, metavar='N', default=2.0, \
help='Scaling factor for marginal likelihood loss of GPs.')
parser.add_argument('--from_ckpt', type=bool, metavar='N', default=False, \
help='Boolen indicating if training or reconstruction should be done using a pre-trained model.')
parser.add_argument('--ckpt_path', type=str, metavar='N', default='', \
help='Path to ckpt with saved model state to be loaded. Only effective if --from_ckpt == True.')
parser.add_argument('--recons_only', type=bool, metavar='N', default=False, \
help='Boolean indicating if trainig is to be skipped. Only use if wanting to inspect reconstruction from intermediate ckpt files.')

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
	fMRI_data = data.FMRIDataset(csv_file = args.csv_file, transform = data.ToTensor())
	model = vae_reg.VAE(num_inducing_pts = args.num_inducing_pts, \
	mll_scale = args.mll_scale)
	if args.from_ckpt == True:
		print('='*40)
		print('Loading model state from: {}'.format(args.ckpt_path))
		model.load_state(filename = args.ckpt_path)
	if not args.recons_only == True:
		model.train_loop(loaders_dict, epochs = args.epochs, test_freq = args.test_freq,\
		save_freq = args.save_freq, save_dir=args.save_dir)
	model.project_latent(loaders_dict, title = "Latent Space plot", split=args.split, save_dir=args.save_dir)
	model.plot_GPs(csv_file=args.csv_file, save_dir=args.save_dir)
	recon.mk_single_volumes(fMRI_data, model, args.csv_file, args.save_dir)
	recon.mk_avg_maps(args.csv_file, model, args.save_dir, mk_motion_maps = False)
	main_end = time.time()
	print('Total model runtime (seconds): {}'.format(main_end - main_start))
