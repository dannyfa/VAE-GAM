"""
Short script to calculate explained variance for each covariate.

This is done on a per subj basis.

Requires that avg maps have been created for all covariates reported!
Be sure to re-run reconstruction routines with mk_motion_maps = True if wanting
to calculate explained variance for motion covariates as well.

Tries to answer question of which covariates included are useful.

Might also add global (i.e., across subjs) explained variance for each covariate (to be seen).
"""

import os, sys
import numpy as np
import nibabel as nib
import re
import argparse
import glob

parser = argparse.ArgumentParser(description='user args')

parser.add_argument('--analysis_dir', type=str, metavar='N', default='', \
help='Path to analysis directory where model individual and avg reconstruction maps are stored.')

args = parser.parse_args()

#find reconstruction directories
#if not found, output an error msg and exit
recons_dir = os.path.join(args.analysis_dir, 'reconstructions')
#if more than 1 sngl vol recon dir exists, pick one with highest epoch #
sngl_vol_recons_dirs = glob.glob(recons_dir + '/' + '*_model_recons')[-1]
if not os.path.exists(sngl_vol_recons_dirs):
    print('No single volume reconstruction directory found!')
    sys.exit()
#if more than 1 avg recon dir exists, picks one with highest epoch #
avg_vol_recons_dirs = glob.glob(recons_dir + '/' + '*_avg_model_recons')[-1]
if not os.path.exists(avg_vol_recons_dirs):
    print('No avg volume reconstruction directory found!')
    sys.exit()

#get subjIDs
RE = re.compile('\Asub-A000*') #regex for finding subjIDs
subj_dirs = os.listdir(sngl_vol_recons_dirs)
subjs = []
for i in range(len(subj_dirs)):
    if RE.search(subj_dirs[i]):
        subjs.append(subj_dirs[i])

maps = ['task', 'x_mot', 'y_mot', 'z_mot', 'pitch_mot', 'yaw_mot', 'roll_mot']

for subj in subjs:
    #get subj full reconstruction avg map
    #this will be used to calc explained variance for all covariates
    frec_avg_path = os.path.join(avg_vol_recons_dirs, subj, 'full_rec_avg.nii')
    frec_avg = np.array(nib.load(frec_avg_path).dataobj)
    print(20*'=')
    print('Subj: {}'.format(subj))
    for map in maps:
        #get avg covariate map for subj
        cov_avg_path = os.path.join(avg_vol_recons_dirs, subj, '{}_avg.nii'.format(map))
        if not os.path.exists(cov_avg_path):
            print('Avg map for covariate {} not found!'.format(map))
            print('Will not report explained variance for {}'.format(map))
            pass
        else:
            #proceed w/ calc for given covariate ...
            cov_avg = np.array(nib.load(cov_avg_path).dataobj)
            numerator, denominator = [], []
            #get subj dir and sub vols dirs
            subj_dir = os.path.join(sngl_vol_recons_dirs, subj)
            subj_vol_dirs = os.listdir(subj_dir)
            #now calc numerator and denominator
            for j in subj_vol_dirs:
                if j[-1] == '1':
                    cov_path, frec_path =  os.path.join(subj_dir, j, 'recon_{}.nii'.format(map)), \
                    os.path.join(subj_dir, j, 'recon_full_rec.nii')
                    cov, frec = np.array(nib.load(cov_path).dataobj), \
                    np.array(nib.load(frec_path).dataobj)
                    num = np.square(cov - cov_avg).sum() #sum here is across voxels!
                    numerator.append(num)
                    den = np.square(frec - frec_avg).sum() #sum here is across voxels!
                    denominator.append(den)
                else:
                    pass
            # now calc subj explained variance for covariate
            # dividivng through number of vols is actually NOT needed here
            # sum is over total # of task volumes in this case
            top = np.array(numerator).mean()
            bottom = np.array(denominator).mean()
            subj_cov_explained_vars =  1 - (bottom - (top/bottom))/bottom
            print('Covariate {} explained variance is: {:.2e}'.format(map, subj_cov_explained_vars))
