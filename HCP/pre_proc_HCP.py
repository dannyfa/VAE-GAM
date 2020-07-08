"""
Script for pre-processing HCP motor task data into useful samples for VAE-reg model
Writes output dset to csv file.
This file should be given as arg to FMRIDataset class.

This is basically same script as for checker set, only w/ small modifications
to satisfy differences in task design and timing...

Major changes:
- Took away demographics info from csv (not used in the end)
- Took away option for different link functions (these are only relevant in controls)

ToDo's:
- Make TR user inputs
- Make StimToNeural more flexible -- i.e., able to read timings from a text file for each subj
- Make finding needed nifti and tsv files less hard-coded

"""
import os, sys
import re
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
import argparse
import scipy.stats
from scipy.stats import gamma # for HRF funct


parser = argparse.ArgumentParser(description='user args for fMRIvae preproc')

parser.add_argument('--data_dir', type=str, metavar='N', default='', \
help='Root dir where nii files and motion .tsv files are located.')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir where csv file to be given as input to FMRIDataset class is saved.')

args = parser.parse_args()

#setting up data_dir
if args.data_dir=='':
    args.data_dir = os.getcwd()
else:
    if not os.path.exists(args.data_dir):
        print('Data dir given does not exist!')
        print('Cannot proceed w/out data!')
        sys.exit()

#setting up save_dir
if args.save_dir == '':
    args.save_dir = os.getcwd()
else:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        pass

#get subjIDs
#total of 9 for this set
#not excluding anyone at this pt
RE = re.compile('\Asub-*')
dirs = os.listdir(args.data_dir)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        subjs.append(dirs[i])
#mk sure we are catching all subjs as desired
print(subjs)

# get paths to pre-processed nii files and to tsv mot files
raw_data_files = []
raw_reg_files = []
for i in range(len(subjs)):
    full_path = os.path.join(args.data_dir, subjs[i])
    for data_file in Path(full_path).rglob('sub-*_preproc_bold_brainmasked_resampled.nii.gz'):
        raw_data_files.append(str(data_file))
    for reg_file in Path(full_path).rglob('sub-*_task-MOTORlr_desc-confounds_regressors_motor_task_analysis.tsv'):
        raw_reg_files.append(str(reg_file))
print(len(raw_data_files))
print(len(raw_reg_files))

#creating raw_df
raw_df = {'nii_files': raw_data_files, 'subjs': subjs, 'regressors': raw_reg_files}
raw_df = pd.DataFrame(raw_df)

#HRF funct.
def hrf(times):
    """ Return values for HRF at given times """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

#altered this to match required timings
#used lh.txt file for sub-101107
#in future, might want something more flexible (i.e., using each subjs time file)
def stimulus_to_neural(vol_times):
    t = vol_times
    res = []
    for i in t:
        if i>=71.5 and i<=83.5:
            task=1
        elif i>=162 and i<=174:
            task=1
        else:
            task=0
        res.append(task)
    return(np.array(res))

#Building final csv file
samples = []
for i in raw_df['subjs']:
    #getting subjid & path to motion regressors
    subjid = i
    #if using fmriprep files
    raw_reg = raw_df.loc[raw_df['subjs'] == i, 'regressors'].iloc[0]
    regressors = pd.read_csv(raw_reg, sep='\t', index_col=False)
    trans_x, trans_y, trans_z = regressors['trans_x'], regressors['trans_y'], regressors['trans_z']
    rot_x, rot_y, rot_z = regressors['rot_x'], regressors['rot_y'], regressors['rot_z']
    #now get fmri dset
    raw_nii = raw_df.loc[raw_df['subjs'] == i, 'nii_files'].iloc[0]
    fmri = np.array(nib.load(raw_nii).dataobj)
    #Get vol time series
    vols = fmri.shape[3]
    #TR should be a user input or something we read from header of nifti files
    TR=0.72
    vol_times = np.arange(1, vols +1) * TR
    neural = stimulus_to_neural(vol_times)
    tr_times = np.arange(0, 20, TR)
    hrf_at_trs = hrf(tr_times)
    #convolve neural stim box-car series w/ HRF
    #take out last value to make arr lengths match
    time_series = np.convolve(neural, hrf_at_trs)
    n_to_remove = len(hrf_at_trs) - 1
    time_series = time_series[:-n_to_remove]
    #finally, build samples...
    for vol in range(vols):
        sample = (subjid, vol, raw_nii, time_series[vol], neural[vol], trans_x[vol], \
        trans_y[vol], trans_z[vol], rot_x[vol], rot_y[vol], rot_z[vol])
        samples.append(sample)
new_df = pd.DataFrame(list(samples), columns=["subjid", "volume #", "nii_path", "task", \
"task_bin", "x", "y", "z", "rot_x", "rot_y", "rot_z"])
save_path = os.path.join(args.save_dir, 'preproc_dset_HCP_wHRF_mot.csv')
new_df.to_csv(save_path)
