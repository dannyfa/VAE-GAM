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

#Calculating grand-avg maps and cons
all_task_vols = []
all_notask_vols = []
for i in range(data.__len__()):
    item = data.__getitem__(i)
    if item['task'] == 1:
        all_task_vols.append(item['volume'])
    else:
        all_notask_vols.append(item['volume'])
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
filepath_avgtask = os.path.join(save_dir, 'grand_task_avg.nii')
nib.save(avg_task_nii, filepath_avgtask)
filepath_avgnotask = os.path.join(save_dir, 'grand_notask_avg.nii')
nib.save(avg_notask_nii, filepath_avgnotask)
filepath_cons = os.path.join(save_dir, 'contrast.nii')
nib.save(cons, filepath_cons)

#Calculating subj task avg maps
for subj in subjs:
    subj_taskvols = []
    subj_notaskvols = []
    for i in range(data.__len__()):
        item = data.__getitem__(i)
        if item['subj'] == subj and item['task']== 1:
            subj_taskvols.append(item['volume'])
        elif item['subj'] == subj and item['task']==0:
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
    filepath_subj_taskavg = os.path.join(save_dir, '{}_taskavg.nii'.format(subj))
    nib.save(subj_taskavg_nii, filepath_subj_taskavg)
    filepath_subj_notaskavg = os.path.join(save_dir, '{}_notaskavg.nii'.format(subj))
    nib.save(subj_notaskavg_nii, filepath_subj_notaskavg)
    filepath_subj_cons= os.path.join(save_dir, '{}_contrast.nii'.format(subj))
    nib.save(subj_cons_nii, filepath_subj_cons)
