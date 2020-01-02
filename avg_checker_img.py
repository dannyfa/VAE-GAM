import os
import numpy as np
import nibabel as nib
import pandas as pd
import DataClass as data

#set up paths to data, ref_nii and saving dir
csv_file = '/home/dfd4/fmri_vae/resampled/preproc_dset.csv'
ref_nii = '/home/dfd4/fmri_vae/resampled/sub-A00057808_resampled.nii.gz'
save_dir = '/home/dfd4/fmri_vae/resampled/avgs'

data = data.FMRIDataset(csv_file = csv_file) # no need to make it into tensor
#getting subj ids and ref nii
dset = pd.read_csv(csv_file)
subjs = dset.subjid.unique().tolist()
input_nifti = nib.load(ref_nii)

#Calculating grand-avg map
all_task_vols = []
for i in range(data.__len__()):
    item = data.__getitem__(i)
    if item['task'] == 1:
        all_task_vols.append(item['volume'])
    else:
        pass
base = np.zeros((41, 49, 35),np.float)
for vol in all_task_vols:
    grand_sum = base + vol
avg = grand_sum/len(all_task_vols)
#write grand avg img to nifti file
avg_task_nii = nib.Nifti1Image(avg, input_nifti.affine, input_nifti.header)
filepath = os.path.join(save_dir, 'grand_task_avg.nii')
nib.save(avg_task_nii, filepath)

#Calculating subj task avg maps
for subj in subjs:
    subj_vols = []
    for i in range(data.__len__()):
        item = data.__getitem__(i)
        if item['subj'] == subj and item['task']== 1:
            subj_vols.append(item['volume'])
        else:
            pass
    #print(subj)
    #print(len(subj_vols))
    subj_base = np.zeros((41, 49, 35),np.float)
    for vol in subj_vols:
        subj_sum = subj_base + vol
    subj_avg = subj_sum/len(subj_vols)
    #print(subj_avg.shape)
    avg_subj_task_nii = nib.Nifti1Image(subj_avg, input_nifti.affine, input_nifti.header)
    filepath = os.path.join(save_dir, '{}_avg.nii'.format(subj))
    nib.save( avg_subj_task_nii, filepath)
