"""
Short script to create Task/NoTask base, cons and full_rec avgs out of model gen data
Does so for entire set and for each subj
Written on Feb 11-2020

ToDo's
Make code less redundant
"""

import os
import numpy as np
import nibabel as nib
import re

#set up paths to data, ref_nii and saving dir
data_root = '/home/dfd4/fmri_vae/new_preproc_dset/1000epochs_HRFConv/model_recons/'
#am using same ref file for entire process
ref_nii = '/home/rachaelwright/fmri_sample_data/checkerboard_and_breathhold/sub-A00057808/ses-NFB2/func/wrsub-A00057808_ses-NFB2_task-CHECKERBOARD_acq-1400_bold.nii'
save_dir = '/home/dfd4/fmri_vae/new_preproc_dset/1000epochs_HRFConv/model_recon_avgs'


#getting input nii file
input_nifti = nib.load(ref_nii)
#getting subjs
RE = re.compile('\Asub-A000*') #regex for finding subjIDs
dirs = os.listdir(data_root)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        subjs.append(dirs[i])

#set up zero arrs for grand avgs
gd_task_base, gd_task_cons,  gd_task_frec= np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float), \
np.zeros((41, 49, 35),np.float)
gd_notask_base, gd_notask_cons, gd_notask_frec = np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float), \
np.zeros((41, 49, 35),np.float)

#now loop through subjs to calc and save subj maps
for subj in subjs:
    subj_dir = os.path.join(data_root, subj)
    subj_vol_dirs = os.listdir(subj_dir)
    #set up zero arr for subj avgs
    subj_task_base, subj_task_cons, subj_task_frec = np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float),\
    np.zeros((41, 49, 35),np.float)
    subj_notask_base, subj_notask_cons, subj_notask_frec = np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float),\
    np.zeros((41, 49, 35),np.float)
    task_vols = 0
    for j in subj_vol_dirs:
        volbase_path, volcons_path, volfrec_path = os.path.join(subj_dir, j, 'recon_base.nii'), os.path.join(subj_dir, j, 'recon_task.nii'), \
        os.path.join(subj_dir, j, 'recon_full_rec.nii')
        volbase, volcons, volfrec = np.array(nib.load(volbase_path).dataobj), np.array(nib.load(volcons_path).dataobj),\
        np.array(nib.load(volfrec_path).dataobj)
        if j[-1] == '1': #this is a task vol
            subj_task_base += volbase
            subj_task_cons += volcons
            subj_task_frec += volfrec
            #uptade counter
            task_vols += 1
        else: #if a notask vol
            subj_notask_base += volbase
            subj_notask_cons += volcons
            subj_notask_frec += volfrec
    #add subj level sums to grand sums
    #task
    gd_task_base += subj_task_base
    gd_task_cons += subj_task_cons
    gd_task_frec += subj_task_frec
    #notask
    gd_notask_base += subj_notask_base
    gd_notask_cons += subj_notask_cons
    gd_notask_frec += subj_notask_frec
    #calc subj task and no task avgs
    notask_vols = len(subj_vol_dirs)-task_vols
    subj_task_baseavg, subj_task_consavg, subj_task_frecavg = subj_task_base/task_vols, subj_task_cons/task_vols, subj_task_frec/task_vols
    subj_notask_baseavg, subj_notask_consavg, subj_notask_frecavg = subj_notask_base/notask_vols, subj_notask_cons/notask_vols, subj_notask_frec/notask_vols
    #mk these into nifti files
    subj_task_baseavg_nii = nib.Nifti1Image(subj_task_baseavg, input_nifti.affine, input_nifti.header)
    subj_task_consavg_nii = nib.Nifti1Image(subj_task_consavg, input_nifti.affine, input_nifti.header)
    subj_task_frecavg_nii = nib.Nifti1Image(subj_task_frecavg, input_nifti.affine, input_nifti.header)
    subj_notask_baseavg_nii = nib.Nifti1Image(subj_notask_baseavg, input_nifti.affine, input_nifti.header)
    subj_notask_consavg_nii = nib.Nifti1Image(subj_notask_consavg, input_nifti.affine, input_nifti.header)
    subj_notask_frecavg_nii = nib.Nifti1Image(subj_notask_frecavg, input_nifti.affine, input_nifti.header)
    #save them
    #task ones
    filepath_subj_task_baseavg = os.path.join(save_dir, '{}_task_baseavg.nii'.format(subj))
    nib.save(subj_task_baseavg_nii, filepath_subj_task_baseavg)
    filepath_subj_task_consavg = os.path.join(save_dir, '{}_task_consavg.nii'.format(subj))
    nib.save(subj_task_consavg_nii, filepath_subj_task_consavg)
    filepath_subj_task_frecavg = os.path.join(save_dir, '{}_task_frecavg.nii'.format(subj))
    nib.save(subj_task_frecavg_nii, filepath_subj_task_frecavg)
    #no task ones
    filepath_subj_notask_baseavg = os.path.join(save_dir, '{}_notask_baseavg.nii'.format(subj))
    nib.save(subj_notask_baseavg_nii, filepath_subj_notask_baseavg)
    filepath_subj_notask_consavg = os.path.join(save_dir, '{}_notask_consavg.nii'.format(subj))
    nib.save(subj_notask_consavg_nii, filepath_subj_notask_consavg)
    filepath_subj_notask_frecavg = os.path.join(save_dir, '{}_notask_frecavg.nii'.format(subj))
    nib.save(subj_notask_frecavg_nii, filepath_subj_notask_frecavg)
#calc grand avgs
total_task_vols = (task_vols)*len(subjs) #since each subj has same num of task vols
total_notask_vols = (98 - task_vols)*len(subjs) #for total of 98 vols per subj
task_baseavg, task_consavg, task_frecavg = gd_task_base/total_task_vols, gd_task_cons/total_task_vols, gd_task_frec/total_task_vols
notask_baseavg, notask_consavg, notask_frecavg = gd_notask_base/total_notask_vols, gd_notask_cons/total_notask_vols, \
gd_notask_frec/total_notask_vols
#now save grand avgs to nii
task_baseavg_nii = nib.Nifti1Image(task_baseavg, input_nifti.affine, input_nifti.header)
task_consavg_nii = nib.Nifti1Image(task_consavg, input_nifti.affine, input_nifti.header)
task_frecavg_nii = nib.Nifti1Image(task_frecavg, input_nifti.affine, input_nifti.header)
notask_baseavg_nii = nib.Nifti1Image(notask_baseavg, input_nifti.affine, input_nifti.header)
notask_consavg_nii = nib.Nifti1Image(notask_consavg, input_nifti.affine, input_nifti.header)
notask_frecavg_nii = nib.Nifti1Image(notask_frecavg, input_nifti.affine, input_nifti.header)
#and save these to actual files
#task ones
filepath_task_baseavg = os.path.join(save_dir, 'task_baseavg.nii')
nib.save(task_baseavg_nii, filepath_task_baseavg)
filepath_task_consavg = os.path.join(save_dir, 'task_consavg.nii')
nib.save(task_consavg_nii, filepath_task_consavg)
filepath_task_frecavg = os.path.join(save_dir, 'task_frecavg.nii')
nib.save(task_frecavg_nii, filepath_task_frecavg)
#no task ones
filepath_notask_baseavg = os.path.join(save_dir, 'notask_baseavg.nii')
nib.save(notask_baseavg_nii, filepath_notask_baseavg)
filepath_notask_consavg = os.path.join(save_dir, 'notask_consavg.nii')
nib.save(notask_consavg_nii, filepath_notask_consavg)
filepath_notask_frecavg = os.path.join(save_dir, 'notask_frecavg.nii')
nib.save(notask_frecavg_nii, filepath_notask_frecavg)
