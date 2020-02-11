import os
import numpy as np
import nibabel as nib
import pandas as pd
import DataClass as data
import vae_reg

#set up paths to data, ref_nii and saving dir
csv_file = '/home/dfd4/fmri_vae/new_preproc_dset/preproc_dset.csv'
ref_nii = '/home/rachaelwright/fmri_sample_data/checkerboard_and_breathhold/sub-A00057808/ses-NFB2/func/wrsub-A00057808_ses-NFB2_task-CHECKERBOARD_acq-1400_bold.nii'
out_dir = '/home/dfd4/fmri_vae/new_preproc_dset/model_recon_avgs'

#create a dset and model objs for reconstruction
data = data.FMRIDataset(csv_file = csv_file, transform = data.ToTensor())
model = vae_reg.VAE()
model.load_state(filename ='/home/dfd4/fmri_vae/new_preproc_dset/100epochs_newpreproc/checkpoint_100.tar')

#get subjids and ref nii
dset = pd.read_csv(csv_file)
subjs = dset.subjid.unique().tolist()
input_nifti = nib.load(ref_nii)

for i in range(len(subjs)):
    #create subj dir to store recons for subj vols
    save_dir = os.path.join(out_dir, subjs[i])
    #Generate model reconstructions for each subj and vol..
    for idx in range(data.__len__()):
        item = data.__getitem__(idx)
        vol_num = dset.iloc[idx,2]
        task = dset.iloc[idx,6]
        ext = str(vol_num) + '_' + str(task) #create a file ext. w/ vol_num and task
        filepath = os.path.join(save_dir, ext)
        os.makedirs(filepath) #mk sure path actually exists  
        model.reconstruct(item, ref_nii= input_nii, save_dir= filepath)
