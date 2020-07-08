"""
Module to create model-based regressor, base and full-reconstruction maps.
Creates:
1) single subject volumes
2) subject level avgs
3) across-subj avgs
Should be called from inside wrapper script using a pre-trained model.
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd

def mk_single_volumes(dataset, model, csv_file, save_dir):
    """
    Creates model's single volume reconstructions for base, regressor and full reconstruction.
    Args:
      dataset :: a torch dataset object (instantiated in wrapper script).
      model :: a VAE_GP model object (instantiated in wrapper script).
      csv_file :: csv with information on dataset.
      save_dir :: root directory where program outputs are written to.
    """
    #get subj_ids and subj reference files from csv
    dset = pd.read_csv(csv_file)
    subjs = dset.subjid.unique().tolist()
    ref_niis = dset.nii_path.unique().tolist()
    #get model epoch #
    ckpt_num = str(model.epoch).zfill(3)
    for i in range(len(subjs)):
        #get corresponding reference nifti files
        subj_ref_nii = ref_niis[i]
        #create subj directory for single volume reconstructions
        subj_dir = os.path.join(save_dir, 'reconstructions', \
        '{}_model_recons'.format(ckpt_num), subjs[i])
        os.makedirs(subj_dir)
        #generate model reconstructions for each vol and each subj
        for idx in range(dataset.__len__()):
            if dset.iloc[idx, 1] == subjs[i]:
                item = dataset.__getitem__(idx)
                vol_num = dset.iloc[idx,2]
                task_bin = dset.iloc[idx,5] #see if worth splitting volumes differently in future?
                ext = 'vol'+ str(vol_num) + '_' + 'task' + str(task_bin)
                filepath = os.path.join(subj_dir, ext)
                os.makedirs(filepath)
                model.reconstruct(item, ref_nii= subj_ref_nii, save_dir= filepath)
            else:
                pass

def mk_avg_maps(csv_file, model, save_dir, mk_motion_maps = False):
    """
    Creates model-based subj-level average and grand average reconstruction maps for
    base, regressors and full reconstruction.
    Maps for motion regressors are ommitted unless otherwise specified.
    Args:
      csv_file :: csv with information on dataset.
      model :: a VAE_GP model object
      save_dir :: root directory where program outputs are written to.
      mk_motion_maps: bool. If 'True', subj and grand avg maps for all 6 motion
      regressors will be reconstructed as well.
    """
    #set up dirs
    ckpt_num = str(model.epoch).zfill(3)
    sngl_vols_dir = os.path.join(save_dir, 'reconstructions', \
    '{}_model_recons'.format(ckpt_num))
    avg_vols_dir = os.path.join(save_dir, 'reconstructions', \
    '{}_avg_model_recons'.format(ckpt_num))
    if not os.path.exists(avg_vols_dir):
        os.makedirs(avg_vols_dir)
    #get ref_nii files and subjs
    dset = pd.read_csv(csv_file)
    ref_niis = dset.nii_path.unique().tolist()
    subjs = dset.subjid.unique().tolist()
    #mk a list of regressors
    maps = ['base', 'task', 'full_rec', 'x_mot', 'y_mot', \
    'z_mot', 'pitch_mot', 'roll_mot', 'yaw_mot']
    if not mk_motion_maps:
        maps = maps[0:3]
    #create dict to hold grand avg maps
    gd_avg_maps = {}
    for l in maps:
        #create name for null maps where task == 0
        null_map = 'null' + '_' + l
        gd_avg_maps[l] = np.zeros((41, 49, 35),np.float)
        gd_avg_maps[null_map] = np.zeros((41, 49, 35),np.float)
        #build single subj avg maps
        for i in range(len(subjs)):
            subj_sngl_vols_dir = os.path.join(sngl_vols_dir, subjs[i])
            subj_vol_dirs = os.listdir(subj_sngl_vols_dir)
            subj_avg_vols_dir = os.path.join(avg_vols_dir, subjs[i])
            if not os.path.exists(subj_avg_vols_dir):
                os.makedirs(subj_avg_vols_dir)
            #create dict for subj-level avg maps
            subj_maps = {}
            subj_maps[l] = np.zeros((41, 49, 35),np.float)
            subj_maps[null_map] = np.zeros((41, 49, 35),np.float)
            #init counter for task volumes
            task_vols = 0
            for k in subj_vol_dirs:
                vol_path = os.path.join(subj_sngl_vols_dir, k, 'recon_{}.nii'.format(l))
                vol = np.array(nib.load(vol_path).dataobj)
                if k[-1] == '1':
                    subj_maps[l] += vol
                    task_vols += 1
                else:
                    subj_maps[null_map] += vol
            #compute subj avg for jth regressor
            notask_vols = len(subj_vol_dirs)-task_vols
            subj_maps[l] /=  task_vols
            subj_maps[null_map] /= notask_vols
            #save subj-level maps
            _save_map(subj_maps[l], ref_niis[i], subj_avg_vols_dir, l)
            _save_map(subj_maps[null_map], ref_niis[i], subj_avg_vols_dir, null_map)
            #update gd avg maps
            gd_avg_maps[l] += subj_maps[l]
            gd_avg_maps[null_map] += subj_maps[null_map]
        #calc grand avg maps
        gd_avg_maps[l] /= len(subjs)
        gd_avg_maps[null_map] /= len(subjs)
        #save grand maps
        #am using zeroth nifti file as Reference
        #however, any file in list is ok since subjs are warped to common space
        _save_map(gd_avg_maps[l], ref_niis[0], avg_vols_dir, l)
        _save_map(gd_avg_maps[null_map], ref_niis[0], avg_vols_dir, null_map)

def _save_map(map, reference, save_dir, ext):
    """
    Helper function for mk_avg_maps
    Takes an array corresponding to a regressor map, a reference nifti file and
    a saving directory and outputs a saved nifti file corresponding to regressor map.
    Only needed b/c I am using nibabel and nifti format.
    This might be taken out entirely in future if we decide to use other libraries or
    file formats like  hdr.
    """
    ref = nib.load(reference)
    nii = nib.Nifti1Image(map, ref.affine, ref.header)
    path = os.path.join(save_dir, '{}_avg.nii'.format(ext))
    nib.save(nii, path)
