"""
Short script to reconstruct all vols from all subjs using a pre-trained model
Needs more flexiblity and some polishing in code
But otherwise works fine
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import DataClass as data
import vae_reg

#set up paths to data, ref_nii and saving dir
csv_file = '/home/dfd4/fmri_vae/new_preproc_dset/preproc_dset_wHRF_mot.csv'
ref_nii = '/home/rachaelwright/fmri_sample_data/checkerboard_and_breathhold/sub-A00057808/ses-NFB2/func/wrsub-A00057808_ses-NFB2_task-CHECKERBOARD_acq-1400_bold.nii'
out_dir = '/home/dfd4/fmri_vae/new_preproc_dset/10000_epsilon_motreg/1000epochs_model_recons'
task_init = '/home/dfd4/fmri_vae/SPM_GLM_avgs/avg_beta_cons1.nii'

#create a dset and model objs for reconstruction
data = data.FMRIDataset(csv_file = csv_file, transform = data.ToTensor())
model = vae_reg.VAE(task_init = task_init)
model.load_state(filename ='/home/dfd4/fmri_vae/new_preproc_dset/10000_epsilon_motreg/checkpoint_1000.tar')

#get subjids and ref nii
dset = pd.read_csv(csv_file)
subjs = dset.subjid.unique().tolist()
input_nifti = nib.load(ref_nii)

for i in range(len(subjs)):
    #create subj dirs to store recons for individual subj vols
    save_dir = os.path.join(out_dir, subjs[i])
    os.makedirs(save_dir)
    #Generate model reconstructions for each vol and each subj
    for idx in range(data.__len__()):
        if dset.iloc[idx, 1] == subjs[i]:
            item = data.__getitem__(idx)
            vol_num = dset.iloc[idx,2]
            task_bin = dset.iloc[idx,7] # changed for task binary var
            ext = str(vol_num) + '_' + str(task_bin) #create a file ext. w/ vol_num and task
            filepath = os.path.join(save_dir, ext)
            os.makedirs(filepath)
            model.reconstruct(item, ref_nii= ref_nii, save_dir= filepath)
        else:
            pass
