"""
Script for pre-processing data into useful samples for VAE-reg model
Writes output dset to csv file.
This file should be given as arg to FMRIDataset class.
Jan 2020
"""
#get dependencies
import os, sys
import re
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
import argparse
from sklearn import preprocessing # for norm step
import scipy.stats
from scipy.stats import gamma # for HRF funct

parser = argparse.ArgumentParser(description='user args for fMRIvae preproc')

parser.add_argument('--data_dir', type=str, metavar='N', default='', \
help='Root dir where nii files and .tsv file are located')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir where csv file to be given as input to FMRIDataset class is saved.')

args = parser.parse_args()

#setting up data_dir
#still needs additional protection for not finding subjs or expected files ....
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
#excluded sub-A00058952 due to high voxel intensity vals
#might re-eval using it if looks ok after new preprocessing...
RE = re.compile('\Asub-A000*') #regex for finding subjIDs
dirs = os.listdir(args.data_dir)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        if 'sub-A00058952' in dirs[i]:
            pass
        else:
            subjs.append(dirs[i])
#print(len(subjs))

#get paths to pre-processed nii files
raw_data_files = []
for i in range(len(subjs)):
    full_path = os.path.join(args.data_dir, subjs[i])
    for data_file in Path(full_path).rglob('wrsub-A000*CHECKERBOARD_acq-1400_bold.nii'):
        raw_data_files.append(str(data_file))
#print(len(raw_data_files))

#getting age and gender for subjs in data_dir
demos_file = os.path.join(args.data_dir,'participants.tsv') #also hardcoded for now
subj_demos = pd.read_csv(demos_file, sep='\t', index_col=False)
#getting stripped subjIDs
striped = [subjs[i][4:] for i in range(len(subjs))]
#getting age and sex lists
age_list = []
sex_list = []
for i in striped:
    age = subj_demos.loc[subj_demos['participant_id'] == i, 'age'].iloc[0]
    age_list.append(age)
    sex = subj_demos.loc[subj_demos['participant_id'] == i, 'sex'].iloc[0]
    sex_list.append(sex)
#convert sex to binary inputs
sex_bin = [1 if i=='FEMALE' else 0 for i in sex_list]
#print(len(sex_bin))
#normalize age values
age_array = np.array(age_list)
normalized_age = preprocessing.normalize([age_array]).tolist()[0]
#print(len(normalized_age))

#creating raw_df
raw_df = {'nii_files': raw_data_files, 'subjs': subjs, 'age': normalized_age, 'sex': sex_bin}
raw_df = pd.DataFrame(raw_df)

#def HRF funct.
#this yields a function form that is very similar to Glover's canonical HRF
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

#get hrf vals for each block, assuming sampling rate==TR
#TR is 1.4s and block len is 20s for checker dset
TR=1.4
tr_times = np.arange(0, 20, TR)
hrf_at_trs = hrf(tr_times)

#set up function to convert vol acquisition times to stimulus responses
#should yield a boxcar like function w/ box length = block duration
def stimulus_to_neural(vol_times):
    t = vol_times//20
    res = []
    for i in t:
        if i==0:
            task=0
        elif i%2==0:
            task=0
        elif i%2!=0:
            task=1
        res.append(task)
    return(np.array(res))

#Building csv file
samples = []
for i in raw_df['subjs']:
    #getting subjid, age, sex, path to raw nii files
    subjid = i
    age = raw_df.loc[raw_df['subjs'] == i, 'age'].iloc[0]
    sex = raw_df.loc[raw_df['subjs'] == i, 'sex'].iloc[0]
    raw_nii = raw_df.loc[raw_df['subjs'] == i, 'nii_files'].iloc[0]
    # load fmri dset
    fmri = np.array(nib.load(raw_nii).dataobj)
    #Get vol stimulus response vals to convolve w/ HRF
    vols = fmri.shape[3]
    vol_times = np.arange(1, vols +1) * TR
    neural = stimulus_to_neural(vol_times)
    #perform convolution to get real-valued task values
    convolved = np.convolve(neural, hrf_at_trs)
    #get last/edge element out
    n_to_remove = len(hrf_at_trs) - 1
    convolved = convolved[:-n_to_remove]
    #build samples
    for vol in range(vols):
        sample = (subjid, vol, raw_nii, age, sex, convolved[vol], neural[vol])
        samples.append(sample)
new_df = pd.DataFrame(list(samples), columns=["subjid","volume #", "nii_path", "age", "sex", "task", "task_bin"])
save_path = os.path.join(args.save_dir, 'preproc_dset.csv')
new_df.to_csv(save_path)
