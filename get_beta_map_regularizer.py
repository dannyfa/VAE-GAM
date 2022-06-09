"""
Script to compute beta maps for regularization term from FSL's design matrix outputs.

Should be run along with other preprocessing scripts and before model training.

---------------
Modifications:
- Current script constructs OLS estimate maps for each subj (i.e., across multiple runs)
- And then subj maps are averaged over to yield final regularization maps
- Does NOT incorporate sex analysis map (YET)
- Rsampling is done in a separate script -- to be run in LaBar server, with afni install.

To Do's:
- Add additional structure to compute sex/gender analysis reg. map
- Implement running version of OLS estimate (this is prolly better than averaging procedure ^^^)
- Do rsampling in python (and without having to rely on afni??)

"""

import re
import os, sys
import nibabel as nib
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import utils
from utils import str2bool

parser = argparse.ArgumentParser(description='user args for beta map regularization script.')

parser.add_argument('--root_dir', type=str, metavar='N', default='', \
help='Root directory containing subdirs for each subject and for .feat FSL analysis for each subject.')
parser.add_argument('--output_dir', type=str, metavar='N', default='', \
help='Output where resulting .csv file with beta maps should be written to.')
parser.add_argument('--data_dims', type=int, metavar='N', default='', nargs='+', \
help='Original dimensions for fMRI data being processed. Should be in order x, y, z, time.')
parser.add_argument('--add_sex_map', type=str2bool, nargs='?', const=True, default=False, \
help='Boolean flag indicating if regularizer map for sex/gender contrast should also be created.')
parser.add_argument('--sex_covars_map', type=str, metavar='N', default='', \
help='Full path to sex covariate cope map produced in higher level analysis in FSL.')

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
RE = re.compile('\Asub-EM0*') #test this regex
dirs = os.listdir(args.root_dir)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        subjs.append(dirs[i])

#make sure we found some subjs!
assert len(subjs)!=0, 'Could not find any subjID matching expected pattern on root dir.'

#now get corresponding .feat directories for these subjs
#these should be 4 (one per run) per subj
feat_dirs = {}
for i in range(len(subjs)):
    feat_dirs[subjs[i]] = []
    subj_dir = os.path.join(args.root_dir, subjs[i])
    for feat_dir in Path(subj_dir).rglob('First_level_run*.feat'):
        feat_dirs[subjs[i]].append(str(feat_dir))

#mk sure subj keys have .feat dirs for 4 runs
for key in list(feat_dirs.keys()):
    assert len(feat_dirs[key])==4, '{} does not have correct # .feat directories. Expected 4 dirs but found {}'.format(key, len(feat_dirs[key]))


#get individual subj maps
#and avg them
all_subj_OLS_maps = []
for i in range(len(subjs)):
    subj_filt_data = utils.get_all_runs_data(feat_dirs, i)
    subj_dms = utils.get_all_runs_dms(feat_dirs, i)
    subj_OLS_sln = utils.get_OLS_sln(subj_dms, subj_filt_data)
    all_subj_OLS_maps.append(subj_OLS_sln)
all_subj_OLS_maps = np.array(all_subj_OLS_maps)
print(all_subj_OLS_maps.shape)
avg_OLS_maps = np.mean(all_subj_OLS_maps, axis=0)
print(avg_OLS_maps.shape)


#add sex covariate to our beta_maps
#this probably won't change MUCH but am leaving it commented for now.
#sex_map = np.array(nib.load(args.sex_covars_map).dataobj)
#sex_map = np.expand_dims(sex_map.flatten(), axis=0)
#beta_maps = np.concatenate([beta_maps, sex_map], axis=0)

#now scale these out
scld_maps = utils.scale_beta_maps(avg_OLS_maps)
#and save them
beta_maps_df = pd.DataFrame(scld_maps.T, columns = ['flow', 'reappraisal', 'distancing', 'x', 'y', 'z', 'xrot', 'yrot', 'zrot', 'sex'])
beta_maps_df.to_csv(os.path.join(args.output_dir, 'scld_GLM_beta_maps.csv'))
