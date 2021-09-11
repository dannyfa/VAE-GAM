"""
Script to compute beta maps for regularization term from FSL's design matrix outputs.

Should be run along with other preprocessing scripts and before model training.
"""

import re
import os, sys
import nibabel as nib
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import utils

parser = argparse.ArgumentParser(description='user args for beta map regularization script.')

parser.add_argument('--root_dir', type=str, metavar='N', default='', \
help='Root directory containing subdirs for each subject and for .feat FSL analysis for each subject.')
parser.add_argument('--output_dir', type=str, metavar='N', default='', \
help='Output where resulting .csv file with beta maps should be written to.')
parser.add_argument('--data_dims', type=int, metavar='N', default='', nargs='+', \
help='Dimensions for fMRI data being processed. Should be in order x, y, z, time.')

args = parser.parse_args()

data_dims = args.data_dims

#make sure root dir exists
if not os.path.exists(args.root_dir):
    print('Root dir given does not exist!')
    print('Cannot proceed w/out data!')
    sys.exit()

#make sure we have an output dir to write to
#default is to write to current dir if '' is parsed.
if args.output_dir == '':
    args.output_dir = os.getcwd()
else:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        pass

#first find all subjs
#excluded sub-A00058952 d/t excess movement.
RE = re.compile('\Asub-A000*')
dirs = os.listdir(args.root_dir)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        if 'sub-A00058952' in dirs[i]:
            pass
        else:
            subjs.append(dirs[i])

#make sure we found some subjs!
assert len(subjs)!=0, 'Could not find any subjID matching expected pattern on root dir.'

#now get corresponding .feat directories for these subjs
feat_dirs = []
for i in range(len(subjs)):
    subj_dir = os.path.join(args.root_dir, subjs[i])
    for feat_dir in Path(subj_dir).rglob('*_corrected.feat'):
        feat_dirs.append(str(feat_dir))

#mk sure len(feat_dirs)!=len(subjs)
assert len(subjs)==len(feat_dirs), 'Not all subjs have .feat directories!'

#get filtered data for each subj
all_subjs_data = []
for i in range(len(subjs)):
    subj_filt_data_path = os.path.join(feat_dirs[i], 'filtered_func_data.nii.gz')
    assert os.path.exists(subj_filt_data_path), 'Failed to find filtered data for subj {}'.format(subjs[i])
    subj_filtered_data = np.array(nib.load(subj_filt_data_path).dataobj).reshape(-1, data_dims[3])
    all_subjs_data.append(subj_filtered_data)
filtered_data = np.concatenate(all_subjs_data, axis=1)

#get all FSL design matrices
all_dms = []
for i in range(len(subjs)):
    subj_mat_path = os.path.join(feat_dirs[i], 'design.mat')
    assert os.path.exists(subj_mat_path), 'Failed to find design matrix for subj {}'.format(subjs[i])
    subj_mat = utils.read_design_mat(subj_mat_path)
    task_col = subj_mat[:, 0].reshape((data_dims[3], 1)) #considers task is first col of DM.
    mot_cols = subj_mat[:, -6:] #considers motion params are last 6 columns of DM.
    subj_mat = np.concatenate((task_col, mot_cols), axis=1)
    all_dms.append(subj_mat)
gamma = np.concatenate(all_dms, axis=0)

#compute GLM Least Square solution maps using the above
pseudo_inv = np.linalg.inv(np.matmul(gamma.T, gamma))
pseudo_inv = np.matmul(pseudo_inv, gamma.T)
beta_maps = np.matmul(pseudo_inv, filtered_data.T)

#scale these maps (max scaling)
scld_beta_maps = utils.scale_beta_maps(beta_maps)
#and save them
beta_maps_df = pd.DataFrame(scld_beta_maps.T, columns = ['task', 'x', 'y', 'z', 'xrot', 'yrot', 'zrot'])
beta_maps_df.to_csv(os.path.join(args.output_dir, 'scld_GLM_beta_maps.csv'))
