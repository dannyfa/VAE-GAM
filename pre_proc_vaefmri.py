"""
Script for pre-processing data into useful samples for VAE-reg model
Writes output dset to csv file.
This file should be given as arg to FMRIDataset class.
November 2019
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
import nilearn as nil # for resampling step

parser = argparse.ArgumentParser(description='user args for fMRIvae preproc')

parser.add_argument('--data_dir', type=str, metavar='N', default='', \
help='Root dir where nii files and .tsv file are located')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir where csv file to be given as input to FMRIDataset class is saved.')
parser.add_argument('--ref_nii', type=str, metavar='N', default = '',  \
help='Reference nii file used to resample raw dset. If file not on scripts dir, full path must be given. Assumes ext *.nii.gz')

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

#setting up ref_nii path
if args.ref_nii=='':
    root = os.getcwd()
    ref_nii = os.path.join(root, '*.nii.gz') # looks for any nifti file in cwd. Test this out!!
else:
    pass

#get subjIDs
RE = re.compile('\Asub-A000*') #regex for finding subjIDs
dirs = os.listdir(args.data_dir)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        subjs.append(dirs[i])
    else:
        pass

#get paths to unsampled nii files
raw_data_files = []
for i in range(len(subjs)):
    full_path = os.path.join(args.data_dir, subjs[i])
    for data_file in Path(full_path).rglob('*CHECKERBOARD_acq-1400_bold.nii.gz'):
        raw_data_files.append(str(data_file))

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
#normalize age values
age_array = np.array(age_list)
normalized_age = preprocessing.normalize([age_array]).tolist()[0]

#creating raw_df
raw_df = {'nii_files': raw_data_files, 'subjs': subjs, 'age': normalized_age, 'sex': sex_bin}
raw_df = pd.DataFrame(raw_df)

#Building csv file
samples = []
ref = nib.load(args.ref_nii)
affine = ref.affine
resampled_shape = (41, 49, 35) # shape of Bx, By, Bz axes in resampled.
for i in raw_df['subjs']:
    #getting subjid, age, sex, path to raw nii files
    subjid = i
    age = raw_df.loc[raw_df['subjs'] == i, 'age'].iloc[0]
    sex = raw_df.loc[raw_df['subjs'] == i, 'sex'].iloc[0]
    raw_nii = raw_df.loc[raw_df['subjs'] == i, 'nii_files'].iloc[0]
    try:
        # re-sampling raw nii files and saving them to save_dir
        resampled_nii = nil.image.resample_img(raw_nii, target_affine = affine, target_shape = resampled_shape, \
        interpolation = 'continuous', copy = True, order = 'F')
        resampled_path = os.path.join(args.save_dir, '{}_resampled.nii.gz'.format(subjid))
        nib.nifti1.save(resampled_nii, resampled_path)
        # getting vol # and task condition from rsampled files
        fmri = np.array(nib.load(resampled_path).dataobj) #ok since vols do not change with resampling
        for vol in range(fmri.shape[3]):
            time_on_task = (vol+1)*1.4
            t = time_on_task//20
            if t==0:
                task=0
            elif t%2==0:
                task=0
            elif t%2!=0:
                task=1
            sample = (subjid, vol, resampled_path, age, sex, task)
            samples.append(sample)
    except:
        print('Could not resample fmri data for subj {}'.format(i))
        print ('Excluding subj {} from final dset.'.format(i))
        pass
new_df = pd.DataFrame(list(samples), columns=["subjid","volume #", "rsampled_nii_path", "age", "sex", "task"])
save_path = os.path.join(args.save_dir, 'preproc_dset.csv')
new_df.to_csv(save_path)
