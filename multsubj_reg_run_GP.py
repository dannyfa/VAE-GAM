"""
Wrapper used to train VAE-GAM fMRI model, create volume reconstructions and effect maps,
plot VAE latent space projections and plot covariate gain function posteriors.

See README file on how to use/call this script.
"""

import os, sys
import argparse
import numpy as np
import random
import torch
import time
import DataClass_GP as data
import vae_reg_GP as vae_reg
import build_model_recons as recon
from utils import str2bool

parser = argparse.ArgumentParser(description='user args for vae_gam model')

parser.add_argument('--csv_file', type=str, metavar='N', default='', \
help='Full path to csv file with train dset to be used by DataClass and loaders. This is created by the pre_proc script.')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir where model params, latent projection maps, GP plots and reconstruction files are saved to. Default is to save files to current dir.')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', \
help='Input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',\
help='Number of epochs to train (default: 300)')
parser.add_argument('--seed', type=int, default=1, metavar='S', \
help='Random seed (default: 1)')
parser.add_argument('--save_freq', type=int, default=100, metavar='N', \
help='How many epochs to wait before saving training status.')
parser.add_argument('--test_freq', type=int, default=200, metavar='N', \
help='How many epochs to wait before testing.')
parser.add_argument('--split', type=int, metavar='N', default=98, \
help='Number used to change colors when plotting VAE latent projection. This is # of volumes for each subj -- i.e., color scheme is per subj.')
parser.add_argument('--glm_reg_scale', type=float, metavar='N', default=1.0, \
help='Scaling factor for GLM map regularization term (default: 1)')
parser.add_argument('--glm_maps', type=str, metavar='N', default='', \
help='Path to csv file containing matrix with approximate GLM maps, one per covariate.')
parser.add_argument('--num_inducing_pts', type=int, metavar='N', default=6, \
help='Number of inducing points for each regressor 1D GP.')
parser.add_argument('--gp_kl_scale', type=float, metavar='N', default=10.0, \
help='Scaling factor for KL divergence loss terms coming from linear and non-linear (GP) pieces of gamma.')
parser.add_argument('--from_ckpt', type=str2bool, nargs='?', const=True, default=False, \
help='Boolean flag indicating if training and/or reconstruction should be carried using a pre-trained model state.')
parser.add_argument('--ckpt_path', type=str, metavar='N', default='', \
help='Path to ckpt with saved model state to be loaded. Only effective if --from_ckpt == True.')
parser.add_argument('--recons_only', type=str2bool, nargs='?', const=True, default=False, \
help='Boolean flag indicating if trainig is to be skipped.')

args = parser.parse_args()
torch.manual_seed(args.seed)

#set up saving directory
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
	model = vae_reg.VAE(num_inducing_pts = args.num_inducing_pts, gp_kl_scale = args.gp_kl_scale, \
	glm_reg_scale = args.glm_reg_scale, glm_maps = args.glm_maps, save_dir = args.save_dir)
	if args.from_ckpt == True:
		assert os.path.exists(args.ckpt_path), 'Oops, looks like ckpt file given does NOT exist!'
		print('='*40)
		print('Loading model state from: {}'.format(args.ckpt_path))
		model.load_state(filename = args.ckpt_path)
	if args.recons_only == False:
		model.train_loop(loaders_dict, epochs = args.epochs, test_freq = args.test_freq,\
		save_freq = args.save_freq, save_dir=args.save_dir)
		model.project_latent(loaders_dict, title = "Latent Space plot", split=args.split, save_dir=args.save_dir)
		model.plot_GPs(csv_file=args.csv_file, save_dir=args.save_dir)
		recon.mk_single_volumes(loaders_dict['test'], model, args.csv_file, args.save_dir) #will have to replace test loader for train one soon.
		recon.mk_avg_maps(args.csv_file, model, args.save_dir, mk_motion_maps = True)
	else:
		assert args.from_ckpt==True, 'To choose recons_only option, --from_ckpt needs to be TRUE.'
		model.project_latent(loaders_dict, title = "Latent Space plot", split=args.split, save_dir=args.save_dir)
		model.plot_GPs(csv_file=args.csv_file, save_dir=args.save_dir)
		recon.mk_single_volumes(loaders_dict['test'], model, args.csv_file, args.save_dir) #will have to replace test loader for train one soon.
		recon.mk_avg_maps(args.csv_file, model, args.save_dir, mk_motion_maps = True)
	main_end = time.time()
	print('Total model runtime (seconds): {}'.format(main_end - main_start))
