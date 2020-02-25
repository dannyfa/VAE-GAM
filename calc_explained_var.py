"""
Short script to calculate contrast explained variance per subj
This is done for task vols only

Tries to ans question of whether or not model should be able to pick up task effect

Feb 2020
"""

import os
import numpy as np
import nibabel as nib
import re

#pass on data dirs
#will look for avg maps and single vol maps under these
avg_dir = '/home/dfd4/fmri_vae/new_preproc_dset/1000epochs_HRFConv/model_recon_avgs'
recon_dir = '/home/dfd4/fmri_vae/new_preproc_dset/1000epochs_HRFConv/model_recons'

#get subjIDs
#getting subjs
RE = re.compile('\Asub-A000*') #regex for finding subjIDs
dirs = os.listdir(recon_dir)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        subjs.append(dirs[i])

for subj in subjs:
    #get subj  avg recon map
    frec_avg_path = os.path.join(avg_dir, '{}_task_frecavg.nii'.format(subj))
    frec_avg = np.array(nib.load(frec_avg_path).dataobj)
    #init numerator and denominator
    num, den = [], []
    #get subj dir and sub vols dirs
    subj_dir = os.path.join(recon_dir, subj)
    subj_vol_dirs = os.listdir(subj_dir)
    #now calc numerator and denominator
    #task_vols = 0
    for j in subj_vol_dirs:
        if j[-1] == '1': #do this only for task vols
            volcons_path, volfrec_path =  os.path.join(subj_dir, j, 'recon_task.nii'), os.path.join(subj_dir, j, 'recon_full_rec.nii')
            volcons, volfrec = np.array(nib.load(volcons_path).dataobj), np.array(nib.load(volfrec_path).dataobj)
            num.append(np.square(volcons).sum())
            diff = frec_avg - volfrec
            den.append(np.square(diff).sum())
            #task_vols += 1
        else:
            pass
    #now get means over num and den
    num_avg = np.array(num).mean()
    den_avg = np.array(den).sum()
    subj_exp_var = num_avg/den_avg
    print('Cons explained variance for subj {} is: {}'.format(subj, subj_exp_var))
