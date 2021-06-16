"""
Script for pre-processing data into useful samples for VAE-GP model.
Writes output dset to csv file. This file should be given as arg to the FMRIDataset class.

Can be used for either checker dataset or for control experiments.
Main difference between these two is on StimToNeural function (see cmts below).

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

parser = argparse.ArgumentParser(description='user args for fMRIvae preproc')

parser.add_argument('--data_dir', type=str, metavar='N', default='', \
help='Root dir where nifty files are located')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir where csv file to be given as input to FMRIDataset class is saved.')
parser.add_argument('--control', type=str2bool, nargs='?', const=True, default=False, \
help='Boolean flag indicating if csv file created is for running experiment on controls.')
parser.add_argument('--link_function', type=str, metavar='N', default='simple_ts', \
help='Link function for used. Simple_ts default leaves sequence as original binary task design/stim.')
parser.add_argument('--nii_file_pattern', type=str, metavar='N', default='sub-A000*_preproc_bold_brainmasked_resampled.nii.gz', \
help='General pattern for filenames of nifty files to be used. Can contain any wildcard that glob and rgob can handle.')
parser.add_argument('--mot_file_pattern', type=str, metavar='N', \
default='sub-A000*_ses-NFB2_task-CHECKERBOARD_acq-1400_desc-confounds_regressors.tsv', \
help='General pattern for filenames of motion files to be used. Can contain any wildcard that glob and rgob can handle.')


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

assert args.link_function in ['simple_ts', 'normal_hrf'], 'Link function given is NOT supported.'
csv_name_suffix = '_chkr_{}.csv'.format(args.link_function)

if args.control == True:
    csv_name_suffix = '_large3_control_{}.csv'.format(args.link_function)

#get subjIDs
#excluded sub-A00058952 due to high voxel intensity vals && excess movement.
RE = re.compile('\Asub-A000*')
dirs = os.listdir(args.data_dir)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        if 'sub-A00058952' in dirs[i]:
            pass
        else:
            subjs.append(dirs[i])

# get paths to pre-processed nifty files and to .tsv mot files
raw_data_files = []
raw_reg_files = []
for i in range(len(subjs)):
    full_path = os.path.join(args.data_dir, subjs[i])
    for data_file in Path(full_path).rglob(args.nii_file_pattern):
        raw_data_files.append(str(data_file))
    for reg_file in Path(full_path).rglob(args.mot_file_pattern):
        raw_reg_files.append(str(reg_file))

#creating raw_df
raw_df = {'nii_files': raw_data_files, 'subjs': subjs, 'regressors': raw_reg_files}
raw_df = pd.DataFrame(raw_df)

#Building final csv file
samples = []
for i in raw_df['subjs']:
    subjid = i
    raw_reg = raw_df.loc[raw_df['subjs'] == i, 'regressors'].iloc[0]
    regressors = pd.read_csv(raw_reg, sep='\t', index_col=False)
    trans_x, trans_y, trans_z = regressors['trans_x'], regressors['trans_y'], regressors['trans_z']
    rot_x, rot_y, rot_z = regressors['rot_x'], regressors['rot_y'], regressors['rot_z']
    #now get fmri dset
    raw_nii = raw_df.loc[raw_df['subjs'] == i, 'nii_files'].iloc[0]
    fmri = np.array(nib.load(raw_nii).dataobj)
    #Get vol time series
    vols = fmri.shape[3]
    TR=1.4
    vol_times = np.arange(1, vols +1) * TR
    #use control stimToneural if control == True
    if args.control:
        neural = utils.control_stimulus_to_neural(vol_times)
    else:
        neural = utils.stimulus_to_neural(vol_times)

    if args.link_function == 'simple_ts':
        time_series = neural

    else:
        time_series = utils.hrf_convolve(neural)

    #build samples...
    for vol in range(vols):
        sample = (subjid, vol, raw_nii, time_series[vol], trans_x[vol], \
        trans_y[vol], trans_z[vol], rot_x[vol], rot_y[vol], \
        rot_z[vol])
        samples.append(sample)

new_df = pd.DataFrame(list(samples), columns=["subjid","volume #", "nii_path", \
"task", "x", "y", "z", "rot_x", "rot_y", "rot_z"])
#z-score motion variables
zscored_df = utils.zscore(new_df)
ts = datetime.datetime.now().date()
csv_name = 'preproc_dset_zscored_' + ts.strftime('%m_%d_%Y') + csv_name_suffix
save_path = os.path.join(args.save_dir, csv_name)
zscored_df.to_csv(save_path)
