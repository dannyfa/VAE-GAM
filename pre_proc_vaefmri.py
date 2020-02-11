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

#Building csv file
samples = []
for i in raw_df['subjs']:
    #getting subjid, age, sex, path to raw nii files
    subjid = i
    age = raw_df.loc[raw_df['subjs'] == i, 'age'].iloc[0]
    sex = raw_df.loc[raw_df['subjs'] == i, 'sex'].iloc[0]
    raw_nii = raw_df.loc[raw_df['subjs'] == i, 'nii_files'].iloc[0]
    # getting vol # and task condition from nii  files
    fmri = np.array(nib.load(raw_nii).dataobj)
    for vol in range(fmri.shape[3]):
        time_on_task = (vol+1)*1.4
        t = time_on_task//20
        if t==0:
            task=0
        elif t%2==0:
            task=0
        elif t%2!=0:
            task=1
        sample = (subjid, vol, raw_nii, age, sex, task)
        samples.append(sample)
new_df = pd.DataFrame(list(samples), columns=["subjid","volume #", "nii_path", "age", "sex", "task"])
save_path = os.path.join(args.save_dir, 'preproc_dset.csv')
new_df.to_csv(save_path)
