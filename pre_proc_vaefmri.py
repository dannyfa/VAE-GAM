"""
Script for pre-processing EMERALD data into useful samples for VAE-GAM model.
Writes output dset to csv file. This file should be given as arg to the FMRIDataset class.

Made adaptions to account for event-related timing/design.
For now am not adding functionality to add control stims or sex  covariate to this file.
Adding sex as a covariate will be a next stage of this analysis.
"""

import utils
from utils import str2bool
import os, sys
import re
import datetime
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
import argparse
from sklearn import preprocessing

parser = argparse.ArgumentParser(description='user args for VAE-GAM preprocessing script.')

parser.add_argument('--data_dir', type=str, metavar='N', default='', \
help='Root dir where nifty (image) files are located.')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir where output from preprocessing script should be saved to.')
parser.add_argument('--set_tag', type=str, metavar='N', default='TRAIN', \
help='Str indicating which data set (TRAIN, TEST or VAL) this csv file refers to. Used in name of output file.')
parser.add_argument('--nii_file_pattern', type=str, metavar='N', default='sub-EM*_emoreg_run*_preproc_short_tempfilt_smooth_brain_rsampled.nii.gz', \
help='General pattern for filenames of nifti files to be used. Can contain any wildcard that glob and rgob can handle.')
parser.add_argument('--mot_file_pattern', type=str, metavar='N', \
default='sub-EM*_ses-day*_task-emoreg_run-0*_desc-confounds_regressors.tsv', \
help='General pattern for filenames of motion files to be used. Can contain any wildcard that glob and rgob can handle.')
parser.add_argument('--event_timing_file_pattern', type=str, metavar='N', \
default='sub-EM*_ses-day*_task-emoreg_run-0*_events.tsv', \
help='General pattern for filenames containing event timing info for each run.')

args = parser.parse_args()

if args.data_dir=='':
    args.data_dir = os.getcwd()
else:
    if not os.path.exists(args.data_dir):
        print('Data dir given does not exist!')
        print('Cannot proceed w/out data!')
        sys.exit()


if args.save_dir == '':
    args.save_dir = os.getcwd()
else:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        pass

csv_name_suffix = '_{}_EMERALD_simple_ts.csv'.format(args.set_tag)

#first find all subjs
RE = re.compile('\Asub-EM0*')
dirs = os.listdir(os.path.join(args.data_dir, 'fmriprep', '{}_set'.format(args.set_tag)))
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        subjs.append(dirs[i])

raw_df = {}
for s in subjs:
    #construct subj dict
    #make sure to sort info for diff runs out ...
    raw_df[s] = {}
    raw_df[s]['nii_files'] = []
    raw_df[s]['mot_files'] = []
    raw_df[s]['event_files'] = []
    fmriprep_path = os.path.join(args.data_dir, 'fmriprep', '{}_set'.format(args.set_tag), s)
    BIDS_path = os.path.join(args.data_dir, 'BIDS', '{}_set'.format(args.set_tag), s)
    for data_file in Path(fmriprep_path).rglob(args.nii_file_pattern):
        raw_df[s]['nii_files'].append(str(data_file))
    raw_df[s]['nii_files'].sort()
    for motreg_file in Path(fmriprep_path).rglob(args.mot_file_pattern):
        raw_df[s]['mot_files'].append(str(motreg_file))
    raw_df[s]['mot_files'].sort()
    for event_file in Path(BIDS_path).rglob(args.event_timing_file_pattern):
        raw_df[s]['event_files'].append(str(event_file))
    raw_df[s]['event_files'].sort()

#now loop through all subjs
#and get info we want per subj, run from their respective dictionaries
all_samples = []
for s in subjs:
    for r in range(4):
        #read mot file
        run_mot_file = raw_df[s]['mot_files'][r]
        run_mot_regressors = pd.read_csv(run_mot_file, sep='\t', index_col=False)
        #cut out entries corresponding to 4 cutout TRs
        run_mot_regressors.drop(run_mot_regressors.index[0:4], inplace=True)
        #get motion covariates for run
        trans_x, trans_y, trans_z = run_mot_regressors['trans_x'], \
        run_mot_regressors['trans_y'], run_mot_regressors['trans_z']
        rot_x, rot_y, rot_z = run_mot_regressors['rot_x'], \
        run_mot_regressors['rot_y'], run_mot_regressors['rot_z']
        #parse event timing file
        nii_file = np.array(nib.load(raw_df[s]['nii_files'][r]).dataobj)
        run_event_codes = utils.parse_run_event_times(raw_df[s]['event_files'][r], nii_file.shape[3])
        #get OH time series for 3 task events we want
        flow_OH = np.where(run_event_codes=='FLOW', 1, 0)
        distract_OH = np.where(run_event_codes=='DISTRACT', 1, 0)
        reappraisal_OH = np.where(run_event_codes=='REAPPRAISE', 1, 0)
        #now construct run samples
        #and append them to general samples
        for v in range(nii_file.shape[3]):
            sample = (s, v, raw_df[s]['nii_files'][r], flow_OH[v], distract_OH[v], \
            reappraisal_OH[v], trans_x.iloc[v], trans_y.iloc[v], trans_z.iloc[v], \
            rot_x.iloc[v], rot_y.iloc[v], rot_z.iloc[v])
            all_samples.append(sample)

all_samples = np.array(all_samples)

#construct final file and zscore continuous covariates
new_df = pd.DataFrame(list(all_samples), columns=["subjid","volume #", "nii_path", "flow", \
"distraction", "reappraisal", "x", "y", "z", "rot_x", "rot_y", "rot_z"])
#z-score motion variables
zscored_df = utils.zscore(new_df)
ts = datetime.datetime.now().date()
csv_name = 'preproc_dset_zscored_' + ts.strftime('%m_%d_%Y') + csv_name_suffix
save_path = os.path.join(args.save_dir, csv_name)
zscored_df.to_csv(save_path)
