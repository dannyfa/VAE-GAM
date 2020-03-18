"""
Short script to create Task/NoTask base, contrasts and full_rec avgs out of model gen data
Does so for entire set and for each subj
Orig written on Feb 11-2020

ToDo's
Make code less redundant
"""

import os
import numpy as np
import nibabel as nib
import re

#set up paths to data, ref_nii and saving dir
data_root = '/home/dfd4/fmri_vae/new_preproc_dset/10000_epsilon_motreg/1000epochs_model_recons'
#am using same ref file for entire process
ref_nii = '/home/rachaelwright/fmri_sample_data/checkerboard_and_breathhold/sub-A00057808/ses-NFB2/func/wrsub-A00057808_ses-NFB2_task-CHECKERBOARD_acq-1400_bold.nii'
save_dir = '/home/dfd4/fmri_vae/new_preproc_dset/10000_epsilon_motreg/1000epochs_model_recon_avgs'


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
#similarly for mot parameters
gd_task_xmot, gd_task_ymot,  gd_task_zmot= np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float), \
np.zeros((41, 49, 35),np.float)
gd_task_pitchmot, gd_task_rollmot,  gd_task_yawmot= np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float), \
np.zeros((41, 49, 35),np.float)

gd_notask_base, gd_notask_cons, gd_notask_frec = np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float), \
np.zeros((41, 49, 35),np.float)
gd_notask_xmot, gd_notask_ymot,  gd_notask_zmot= np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float), \
np.zeros((41, 49, 35),np.float)
gd_notask_pitchmot, gd_notask_rollmot,  gd_notask_yawmot= np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float), \
np.zeros((41, 49, 35),np.float)

#now loop through subjs to calc and save subj maps
for subj in subjs:
    subj_dir = os.path.join(data_root, subj)
    subj_vol_dirs = os.listdir(subj_dir)
    #set up zero arr for subj avgs
    subj_task_base, subj_task_cons, subj_task_frec = np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float),\
    np.zeros((41, 49, 35),np.float)
    #similarly for mot params
    subj_task_xmot, subj_task_ymot, subj_task_zmot = np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float),\
    np.zeros((41, 49, 35),np.float)
    subj_task_pitchmot, subj_task_rollmot, subj_task_yawmot = np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float),\
    np.zeros((41, 49, 35),np.float)

    subj_notask_base, subj_notask_cons, subj_notask_frec = np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float),\
    np.zeros((41, 49, 35),np.float)
    #similarly for mot params
    subj_notask_xmot, subj_notask_ymot, subj_notask_zmot = np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float),\
    np.zeros((41, 49, 35),np.float)
    subj_notask_pitchmot, subj_notask_rollmot, subj_notask_yawmot = np.zeros((41, 49, 35),np.float), np.zeros((41, 49, 35),np.float),\
    np.zeros((41, 49, 35),np.float)

    task_vols = 0
    for j in subj_vol_dirs:
        volbase_path, volcons_path, volfrec_path = os.path.join(subj_dir, j, 'recon_base.nii'), os.path.join(subj_dir, j, 'recon_task.nii'), \
        os.path.join(subj_dir, j, 'recon_full_rec.nii')
        #now for mot parameters
        volxmot_path, volymot_path, volzmot_path = os.path.join(subj_dir, j, 'recon_x_mot.nii'), os.path.join(subj_dir, j, 'recon_y_mot.nii'), \
        os.path.join(subj_dir, j, 'recon_z_mot.nii')
        volpitchmot_path, volrollmot_path, volyawmot_path = os.path.join(subj_dir, j, 'recon_pitch_mot.nii'), os.path.join(subj_dir, j, 'recon_roll_mot.nii'), \
        os.path.join(subj_dir, j, 'recon_yaw_mot.nii')
        volbase, volcons, volfrec = np.array(nib.load(volbase_path).dataobj), np.array(nib.load(volcons_path).dataobj),\
        np.array(nib.load(volfrec_path).dataobj)
        volxmot, volymot, volzmot = np.array(nib.load(volxmot_path).dataobj), np.array(nib.load(volymot_path).dataobj),\
        np.array(nib.load(volzmot_path).dataobj)
        volpitchmot, volrollmot, volyawmot = np.array(nib.load(volpitchmot_path).dataobj), np.array(nib.load(volrollmot_path).dataobj),\
        np.array(nib.load(volyawmot_path).dataobj)

        if j[-1] == '1': #this is a task vol
            subj_task_base += volbase
            subj_task_cons += volcons
            subj_task_frec += volfrec
            subj_task_xmot += volxmot
            subj_task_ymot += volymot
            subj_task_zmot += volzmot
            subj_task_pitchmot += volpitchmot
            subj_task_rollmot += volrollmot
            subj_task_yawmot += volyawmot
            #uptade counter
            task_vols += 1
        else: #if a notask vol
            subj_notask_base += volbase
            subj_notask_cons += volcons
            subj_notask_frec += volfrec
            subj_notask_xmot += volxmot
            subj_notask_ymot += volymot
            subj_notask_zmot += volzmot
            subj_notask_pitchmot += volpitchmot
            subj_notask_rollmot += volrollmot
            subj_notask_yawmot += volyawmot
    #add subj level sums to grand sums
    #task
    gd_task_base += subj_task_base
    gd_task_cons += subj_task_cons
    gd_task_frec += subj_task_frec
    gd_task_xmot += subj_task_xmot
    gd_task_ymot += subj_task_ymot
    gd_task_zmot += subj_task_zmot
    gd_task_pitchmot += subj_task_pitchmot
    gd_task_rollmot += subj_task_rollmot
    gd_task_yawmot += subj_task_yawmot

    #notask
    gd_notask_base += subj_notask_base
    gd_notask_cons += subj_notask_cons
    gd_notask_frec += subj_notask_frec
    gd_notask_xmot += subj_notask_xmot
    gd_notask_ymot += subj_notask_ymot
    gd_notask_zmot += subj_notask_zmot
    gd_notask_pitchmot += subj_notask_pitchmot
    gd_notask_rollmot += subj_notask_rollmot
    gd_notask_yawmot += subj_notask_yawmot

    #calc subj task and no task avgs
    notask_vols = len(subj_vol_dirs)-task_vols
    subj_task_baseavg, subj_task_consavg, subj_task_frecavg = subj_task_base/task_vols, subj_task_cons/task_vols, subj_task_frec/task_vols
    subj_task_xmotavg, subj_task_ymotavg, subj_task_zmotavg = subj_task_xmot/task_vols, subj_task_ymot/task_vols, subj_task_zmot/task_vols
    subj_task_pitchmotavg, subj_task_rollmotavg, subj_task_yawmotavg = subj_task_pitchmot/task_vols, subj_task_rollmot/task_vols, subj_task_yawmot/task_vols

    subj_notask_baseavg, subj_notask_consavg, subj_notask_frecavg = subj_notask_base/notask_vols, subj_notask_cons/notask_vols, subj_notask_frec/notask_vols
    subj_notask_xmotavg, subj_notask_ymotavg, subj_notask_zmotavg = subj_notask_xmot/notask_vols, subj_notask_ymot/notask_vols, subj_notask_zmot/notask_vols
    subj_notask_pitchmotavg, subj_notask_rollmotavg, subj_notask_yawmotavg = subj_notask_pitchmot/notask_vols, subj_notask_rollmot/notask_vols, subj_notask_yawmot/notask_vols

    #mk these into nifti files
    subj_task_baseavg_nii = nib.Nifti1Image(subj_task_baseavg, input_nifti.affine, input_nifti.header)
    subj_task_consavg_nii = nib.Nifti1Image(subj_task_consavg, input_nifti.affine, input_nifti.header)
    subj_task_frecavg_nii = nib.Nifti1Image(subj_task_frecavg, input_nifti.affine, input_nifti.header)
    #for mot params
    subj_task_xmotavg_nii = nib.Nifti1Image(subj_task_xmotavg, input_nifti.affine, input_nifti.header)
    subj_task_ymotavg_nii = nib.Nifti1Image(subj_task_ymotavg, input_nifti.affine, input_nifti.header)
    subj_task_zmotavg_nii = nib.Nifti1Image(subj_task_zmotavg, input_nifti.affine, input_nifti.header)
    subj_task_pitchmotavg_nii = nib.Nifti1Image(subj_task_pitchmotavg, input_nifti.affine, input_nifti.header)
    subj_task_rollmotavg_nii = nib.Nifti1Image(subj_task_rollmotavg, input_nifti.affine, input_nifti.header)
    subj_task_yawmotavg_nii = nib.Nifti1Image(subj_task_yawmotavg, input_nifti.affine, input_nifti.header)

    subj_notask_baseavg_nii = nib.Nifti1Image(subj_notask_baseavg, input_nifti.affine, input_nifti.header)
    subj_notask_consavg_nii = nib.Nifti1Image(subj_notask_consavg, input_nifti.affine, input_nifti.header)
    subj_notask_frecavg_nii = nib.Nifti1Image(subj_notask_frecavg, input_nifti.affine, input_nifti.header)
    #for mot params
    subj_notask_xmotavg_nii = nib.Nifti1Image(subj_notask_xmotavg, input_nifti.affine, input_nifti.header)
    subj_notask_ymotavg_nii = nib.Nifti1Image(subj_notask_ymotavg, input_nifti.affine, input_nifti.header)
    subj_notask_zmotavg_nii = nib.Nifti1Image(subj_notask_zmotavg, input_nifti.affine, input_nifti.header)
    subj_notask_pitchmotavg_nii = nib.Nifti1Image(subj_notask_pitchmotavg, input_nifti.affine, input_nifti.header)
    subj_notask_rollmotavg_nii = nib.Nifti1Image(subj_notask_rollmotavg, input_nifti.affine, input_nifti.header)
    subj_notask_yawmotavg_nii = nib.Nifti1Image(subj_notask_yawmotavg, input_nifti.affine, input_nifti.header)

    #save them
    #task ones
    filepath_subj_task_baseavg = os.path.join(save_dir, '{}_task_baseavg.nii'.format(subj))
    nib.save(subj_task_baseavg_nii, filepath_subj_task_baseavg)
    filepath_subj_task_consavg = os.path.join(save_dir, '{}_task_consavg.nii'.format(subj))
    nib.save(subj_task_consavg_nii, filepath_subj_task_consavg)
    filepath_subj_task_frecavg = os.path.join(save_dir, '{}_task_frecavg.nii'.format(subj))
    nib.save(subj_task_frecavg_nii, filepath_subj_task_frecavg)
    #for motion parameters
    filepath_subj_task_xmotavg = os.path.join(save_dir, '{}_task_xmotavg.nii'.format(subj))
    nib.save(subj_task_xmotavg_nii, filepath_subj_task_xmotavg)
    filepath_subj_task_ymotavg = os.path.join(save_dir, '{}_task_ymotavg.nii'.format(subj))
    nib.save(subj_task_ymotavg_nii, filepath_subj_task_ymotavg)
    filepath_subj_task_zmotavg = os.path.join(save_dir, '{}_task_zmotavg.nii'.format(subj))
    nib.save(subj_task_zmotavg_nii, filepath_subj_task_zmotavg)

    filepath_subj_task_pitchmotavg = os.path.join(save_dir, '{}_task_pitchmotavg.nii'.format(subj))
    nib.save(subj_task_pitchmotavg_nii, filepath_subj_task_pitchmotavg)
    filepath_subj_task_rollmotavg = os.path.join(save_dir, '{}_task_rollmotavg.nii'.format(subj))
    nib.save(subj_task_rollmotavg_nii, filepath_subj_task_rollmotavg)
    filepath_subj_task_yawmotavg = os.path.join(save_dir, '{}_task_yawmotavg.nii'.format(subj))
    nib.save(subj_task_yawmotavg_nii, filepath_subj_task_yawmotavg)

    #no task ones
    filepath_subj_notask_baseavg = os.path.join(save_dir, '{}_notask_baseavg.nii'.format(subj))
    nib.save(subj_notask_baseavg_nii, filepath_subj_notask_baseavg)
    filepath_subj_notask_consavg = os.path.join(save_dir, '{}_notask_consavg.nii'.format(subj))
    nib.save(subj_notask_consavg_nii, filepath_subj_notask_consavg)
    filepath_subj_notask_frecavg = os.path.join(save_dir, '{}_notask_frecavg.nii'.format(subj))
    nib.save(subj_notask_frecavg_nii, filepath_subj_notask_frecavg)
    #for motion parameters
    #for motion parameters
    filepath_subj_notask_xmotavg = os.path.join(save_dir, '{}_notask_xmotavg.nii'.format(subj))
    nib.save(subj_notask_xmotavg_nii, filepath_subj_notask_xmotavg)
    filepath_subj_notask_ymotavg = os.path.join(save_dir, '{}_notask_ymotavg.nii'.format(subj))
    nib.save(subj_notask_ymotavg_nii, filepath_subj_notask_ymotavg)
    filepath_subj_notask_zmotavg = os.path.join(save_dir, '{}_notask_zmotavg.nii'.format(subj))
    nib.save(subj_notask_zmotavg_nii, filepath_subj_notask_zmotavg)

    filepath_subj_notask_pitchmotavg = os.path.join(save_dir, '{}_notask_pitchmotavg.nii'.format(subj))
    nib.save(subj_notask_pitchmotavg_nii, filepath_subj_notask_pitchmotavg)
    filepath_subj_notask_rollmotavg = os.path.join(save_dir, '{}_notask_rollmotavg.nii'.format(subj))
    nib.save(subj_notask_rollmotavg_nii, filepath_subj_notask_rollmotavg)
    filepath_subj_notask_yawmotavg = os.path.join(save_dir, '{}_notask_yawmotavg.nii'.format(subj))
    nib.save(subj_notask_yawmotavg_nii, filepath_subj_notask_yawmotavg)

#calc grand avgs
total_task_vols = (task_vols)*len(subjs) #since each subj has same num of task vols
total_notask_vols = (98 - task_vols)*len(subjs) #for total of 98 vols per subj

task_baseavg, task_consavg, task_frecavg = gd_task_base/total_task_vols, gd_task_cons/total_task_vols, gd_task_frec/total_task_vols
task_xmotavg, task_ymotavg, task_zmotavg = gd_task_xmot/total_task_vols, gd_task_ymot/total_task_vols, gd_task_zmot/total_task_vols
task_pitchmotavg, task_rollmotavg, task_yawmotavg = gd_task_pitchmot/total_task_vols, gd_task_rollmot/total_task_vols,\
gd_task_yawmot/total_task_vols


notask_baseavg, notask_consavg, notask_frecavg = gd_notask_base/total_notask_vols, gd_notask_cons/total_notask_vols, \
gd_notask_frec/total_notask_vols
notask_xmotavg, notask_ymotavg, notask_zmotavg = gd_notask_xmot/total_notask_vols, gd_notask_ymot/total_notask_vols, \
gd_notask_zmot/total_notask_vols
notask_pitchmotavg, notask_rollmotavg, notask_yawmotavg = gd_notask_pitchmot/total_notask_vols, \
gd_notask_rollmot/total_notask_vols, gd_notask_yawmot/total_notask_vols

#now save grand avgs to nii
task_baseavg_nii = nib.Nifti1Image(task_baseavg, input_nifti.affine, input_nifti.header)
task_consavg_nii = nib.Nifti1Image(task_consavg, input_nifti.affine, input_nifti.header)
task_frecavg_nii = nib.Nifti1Image(task_frecavg, input_nifti.affine, input_nifti.header)
#for motion parameters
task_xmotavg_nii = nib.Nifti1Image(task_xmotavg, input_nifti.affine, input_nifti.header)
task_ymotavg_nii = nib.Nifti1Image(task_ymotavg, input_nifti.affine, input_nifti.header)
task_zmotavg_nii = nib.Nifti1Image(task_zmotavg, input_nifti.affine, input_nifti.header)
task_pitchmotavg_nii = nib.Nifti1Image(task_pitchmotavg, input_nifti.affine, input_nifti.header)
task_rollmotavg_nii = nib.Nifti1Image(task_rollmotavg, input_nifti.affine, input_nifti.header)
task_yawmotavg_nii = nib.Nifti1Image(task_yawmotavg, input_nifti.affine, input_nifti.header)

notask_baseavg_nii = nib.Nifti1Image(notask_baseavg, input_nifti.affine, input_nifti.header)
notask_consavg_nii = nib.Nifti1Image(notask_consavg, input_nifti.affine, input_nifti.header)
notask_frecavg_nii = nib.Nifti1Image(notask_frecavg, input_nifti.affine, input_nifti.header)
#for motion parameters
notask_xmotavg_nii = nib.Nifti1Image(notask_xmotavg, input_nifti.affine, input_nifti.header)
notask_ymotavg_nii = nib.Nifti1Image(notask_ymotavg, input_nifti.affine, input_nifti.header)
notask_zmotavg_nii = nib.Nifti1Image(notask_zmotavg, input_nifti.affine, input_nifti.header)
notask_pitchmotavg_nii = nib.Nifti1Image(notask_pitchmotavg, input_nifti.affine, input_nifti.header)
notask_rollmotavg_nii = nib.Nifti1Image(notask_rollmotavg, input_nifti.affine, input_nifti.header)
notask_yawmotavg_nii = nib.Nifti1Image(notask_yawmotavg, input_nifti.affine, input_nifti.header)

#and save these to actual files
#task ones
filepath_task_baseavg = os.path.join(save_dir, 'task_baseavg.nii')
nib.save(task_baseavg_nii, filepath_task_baseavg)
filepath_task_consavg = os.path.join(save_dir, 'task_consavg.nii')
nib.save(task_consavg_nii, filepath_task_consavg)
filepath_task_frecavg = os.path.join(save_dir, 'task_frecavg.nii')
nib.save(task_frecavg_nii, filepath_task_frecavg)
#for motion parameters
filepath_task_xmotavg = os.path.join(save_dir, 'task_xmotavg.nii')
nib.save(task_xmotavg_nii, filepath_task_xmotavg)
filepath_task_ymotavg = os.path.join(save_dir, 'task_ymotavg.nii')
nib.save(task_ymotavg_nii, filepath_task_ymotavg)
filepath_task_zmotavg = os.path.join(save_dir, 'task_zmotavg.nii')
nib.save(task_zmotavg_nii, filepath_task_zmotavg)

filepath_task_pitchmotavg = os.path.join(save_dir, 'task_pitchmotavg.nii')
nib.save(task_pitchmotavg_nii, filepath_task_pitchmotavg)
filepath_task_rollmotavg = os.path.join(save_dir, 'task_rollmotavg.nii')
nib.save(task_rollmotavg_nii, filepath_task_rollmotavg)
filepath_task_yawmotavg = os.path.join(save_dir, 'task_yawmotavg.nii')
nib.save(task_yawmotavg_nii, filepath_task_yawmotavg)

#no task ones
filepath_notask_baseavg = os.path.join(save_dir, 'notask_baseavg.nii')
nib.save(notask_baseavg_nii, filepath_notask_baseavg)
filepath_notask_consavg = os.path.join(save_dir, 'notask_consavg.nii')
nib.save(notask_consavg_nii, filepath_notask_consavg)
filepath_notask_frecavg = os.path.join(save_dir, 'notask_frecavg.nii')
nib.save(notask_frecavg_nii, filepath_notask_frecavg)
#for motion parameters
filepath_notask_xmotavg = os.path.join(save_dir, 'notask_xmotavg.nii')
nib.save(notask_xmotavg_nii, filepath_notask_xmotavg)
filepath_notask_ymotavg = os.path.join(save_dir, 'notask_ymotavg.nii')
nib.save(notask_ymotavg_nii, filepath_notask_ymotavg)
filepath_notask_zmotavg = os.path.join(save_dir, 'notask_zmotavg.nii')
nib.save(notask_zmotavg_nii, filepath_notask_zmotavg)

filepath_notask_pitchmotavg = os.path.join(save_dir, 'notask_pitchmotavg.nii')
nib.save(notask_pitchmotavg_nii, filepath_notask_pitchmotavg)
filepath_notask_rollmotavg = os.path.join(save_dir, 'notask_rollmotavg.nii')
nib.save(notask_rollmotavg_nii, filepath_notask_rollmotavg)
filepath_notask_yawmotavg = os.path.join(save_dir, 'notask_yawmotavg.nii')
nib.save(notask_yawmotavg_nii, filepath_notask_yawmotavg)
