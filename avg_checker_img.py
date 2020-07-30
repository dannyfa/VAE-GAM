"""
Script to create global and subj specific avgs maps for:
1) Task
2) noTask
3) contrast = task - notask

TODOs:
1. Get rid of need for a reference nifti file arg
2. Add small method to avoid using multiple lines when saving files

These were already done for newer scripts, adapt implementation from there.
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import DataClass_GP as data
import argparse

parser = argparse.ArgumentParser(description='args for creating avg task, notask and contrast maps.')

parser.add_argument('--csv_file', type=str, metavar='N', default='', \
help='Full path to csv file with raw dset. This is created by the pre_proc script.')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir where outputs (avg task, notask and contrast maps) are saved.')
parser.add_argument('--ref_nii', type=str, default='', metavar='N', \
help='Path to a reference nifti. This is needed to save maps created in nifti format.')

args = parser.parse_args()

data = data.FMRIDataset(csv_file = args.csv_file) # no need to make it into tensor
#getting subj ids and ref nii
dset = pd.read_csv(args.csv_file)
subjs = dset.subjid.unique().tolist()
input_nifti = nib.load(args.ref_nii)

#Calculating grand-avg maps and cons
all_task_vols = []
all_notask_vols = []
for i in range(data.__len__()):
    item = data.__getitem__(i)
    if item['task_bin'] == 1:
        all_task_vols.append(item['volume'])
    else:
        all_notask_vols.append(item['volume'])
#print(len(all_task_vols))
#print(len(all_notask_vols))
base = np.zeros((41, 49, 35),np.float)
for vol in all_task_vols:
    grand_sum = base + vol
task_avg = grand_sum/len(all_task_vols)
for vol in all_notask_vols:
    grand_sum = base + vol
notask_avg = grand_sum/len(all_notask_vols)
cons = task_avg - notask_avg
#write nifti files
avg_task_nii = nib.Nifti1Image(task_avg, input_nifti.affine, input_nifti.header)
avg_notask_nii = nib.Nifti1Image(notask_avg, input_nifti.affine, input_nifti.header)
cons = nib.Nifti1Image(cons, input_nifti.affine, input_nifti.header)
filepath_avgtask = os.path.join(args.save_dir, 'grand_task_avg.nii')
nib.save(avg_task_nii, filepath_avgtask)
filepath_avgnotask = os.path.join(args.save_dir, 'grand_notask_avg.nii')
nib.save(avg_notask_nii, filepath_avgnotask)
filepath_cons = os.path.join(args.save_dir, 'contrast.nii')
nib.save(cons, filepath_cons)

#Calculating subj task avg maps
for subj in subjs:
    subj_taskvols = []
    subj_notaskvols = []
    for i in range(data.__len__()):
        item = data.__getitem__(i)
        if item['subj'] == subj and item['task_bin']== 1:
            subj_taskvols.append(item['volume'])
        elif item['subj'] == subj and item['task_bin']==0:
            subj_notaskvols.append(item['volume'])
        else:
            pass
    subj_base = np.zeros((41, 49, 35),np.float)
    for vol in subj_taskvols:
        subj_sum = subj_base + vol
    subj_taskavg = subj_sum/len(subj_taskvols)
    for vol in subj_notaskvols:
        subj_sum = subj_base + vol
    subj_notaskavg = subj_sum/len(subj_notaskvols)
    subj_cons = subj_taskavg - subj_notaskavg
    #write subj nii files
    subj_taskavg_nii = nib.Nifti1Image(subj_taskavg, input_nifti.affine, input_nifti.header)
    subj_notaskavg_nii = nib.Nifti1Image(subj_notaskavg, input_nifti.affine, input_nifti.header)
    subj_cons_nii = nib.Nifti1Image(subj_cons, input_nifti.affine, input_nifti.header)
    filepath_subj_taskavg = os.path.join(args.save_dir, '{}_taskavg.nii'.format(subj))
    nib.save(subj_taskavg_nii, filepath_subj_taskavg)
    filepath_subj_notaskavg = os.path.join(args.save_dir, '{}_notaskavg.nii'.format(subj))
    nib.save(subj_notaskavg_nii, filepath_subj_notaskavg)
    filepath_subj_cons= os.path.join(args.save_dir, '{}_contrast.nii'.format(subj))
    nib.save(subj_cons_nii, filepath_subj_cons)
