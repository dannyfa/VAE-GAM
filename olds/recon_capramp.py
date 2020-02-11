"""
Script to create create recons at different training stages.
Requires checkpoint files for model at different points in training

December 2019
"""

import os, sys
import argparse
import numpy as np
import glob
import torch
import DataClass as data #w/ FMRIDataClass, trsfm and loaders
import vae_reg

parser = argparse.ArgumentParser(description='user args for calculating recons')

parser.add_argument('--data_dir', type=str, metavar='N', default='', \
help='Dir where model checkpoint files are located.')
parser.add_argument('--csv_file', type=str, metavar='N', default='/home/dfd4/fmri_vae/resampled/preproc_dset.csv', \
help='Full path to csv file with raw dset to used by DataClass and loaders. This is created by the pre_proc script.')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir where recons are saved to. Defaults to saving files in current dir.')
parser.add_argument('--ref_nii', type=str, metavar='N', default='', \
help='Full path to reference nii file to be used for reconstructions.')

args = parser.parse_args()

#set up saving directory specs
if args.save_dir =='':
	args.save_dir = os.getcwd()
if args.save_dir != '' and not os.path.exists(args.save_dir):
	os.makedirs(args.save_dir)
else:
	pass
#getting checkpoint files
chckpt_files = glob.glob(os.path.join(args.data_dir, 'checkpoint_*.tar'))
#getting volume to be reconstructed
#same vol is used for all states for consistency
data = data.FMRIDataset(csv_file = args.csv_file, transform = data.ToTensor())
idx= 18 # making this 18th item in dset for now. Will be user input later. This is a vol with task == 1
item = data.__getitem__(idx)
#getting recons
for file in chckpt_files:
    model = vae_reg.VAE()
    model.load_state(filename = file)
    unique_dir = os.path.join(args.save_dir, file[:-4])
    os.makedirs(unique_dir)
    model.reconstruct(item, ref_nii=args.ref_nii, save_dir=unique_dir)
