"""
Short script to create avg SPM GLM stat map (for comparison) &
avg beta SPM map (for init in model).
Does so for checker effect (#1) and checker>fixation (#4)
"""

import os, sys
import re
from pathlib import Path
import nibabel as nib
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='args for calc SPM GLM avgs')

parser.add_argument('--data_dir', type=str, metavar='N', default='', \
help='Root dir where subj SPM GLM beta and stat maps are.')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir to write out output avg maps.')
parser.add_argument('--ref_nii', type=str, metavar='N', default='', \
help='Ref nii from which hdr and affine are taken. Can be fmriprep pre-processed file from any subj.')

args = parser.parse_args()

#setting up data_dir
if args.data_dir=='':
    args.data_dir = os.getcwd()
else:
    if not os.path.exists(args.data_dir):
        print('Data dir given does not exist!')
        print('Cannot proceed w/out data!')
        sys.exit()

#setting up save_dir
if args.save_dir == '':
    args.save_dir = os.getcwd()
else:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        pass

# get ref nifti
input_nifti = nib.load(args.ref_nii)

#get subjIDs
#excluded sub-A00058952 due to high voxel intensity vals
RE = re.compile('\Asub-A000*') #regex for finding subjIDs
dirs = os.listdir(args.data_dir)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        if 'sub-A00058952' in dirs[i]:
            pass
        else:
            subjs.append(dirs[i])

#get stat_maps and beta_maps for all subj
stat_maps = {}
beta_maps = {}

for i in range(len(subjs)):
    subj_stats = {'cons1':{}, 'cons2':{}, 'cons3':{}, 'cons4':{}, 'cons5':{}}
    full_path = os.path.join(args.data_dir, subjs[i],'ses-NFB2', 'func', \
    '{}_task-preproc_bold_brainmasked_resampled_corrected.feat'.format(subjs[i]))
    #get stat maps
    for stat_map in Path(full_path).rglob('thresh_zstat*.nii.gz'):
        stat_path = str(stat_map)
        num = re.findall(r'\d+', stat_path)[-1]
        stat_map = np.array(nib.load(stat_path).dataobj)
        key = 'cons{}'.format(num)
        try:
            subj_stats[key] = stat_map
        except:
            pass
    stat_maps[subjs[i]] = subj_stats

for i in range(len(subjs)):
    subj_betas = {'cons1':{}, 'cons2':{}, 'cons3':{}, 'cons4':{}, 'cons5':{}}
    full_path = os.path.join(args.data_dir, subjs[i],'ses-NFB2', 'func', \
    '{}_task-preproc_bold_brainmasked_resampled_corrected.feat'.format(subjs[i]))
    #get beta maps
    for cope_file in Path(full_path).rglob('cope*.nii.gz'):
        beta_path = str(cope_file)
        num = re.findall(r'\d+', beta_path)[-1]
        beta_map =  np.array(nib.load(beta_path).dataobj)
        key = 'cons{}'.format(num)
        try:
            subj_betas[key] = beta_map
        except:
            pass
    beta_maps[subjs[i]] = subj_betas

#now compute avg_maps
#loop through num of contrasts/conditions
for i in range (1,6):
    gd_stat_map = np.zeros((41, 49, 35), dtype=np.float32)
    gd_beta_map = np.zeros((41, 49, 35), dtype=np.float32)
    name = 'cons{}'.format(i)
    for subj in subjs:
        gd_stat_map += stat_maps[subj][name]
        gd_beta_map += beta_maps[subj][name]
    #div though len subjs...
    gd_stat_map = gd_stat_map/len(subjs)
    gd_beta_map = gd_beta_map/len(subjs)
    #create nii objs
    gd_stat_map_nii =  nib.Nifti1Image(gd_stat_map, input_nifti.affine, input_nifti.header)
    gd_beta_map_nii =  nib.Nifti1Image(gd_beta_map, input_nifti.affine, input_nifti.header)
    #write outputs
    stat_out_path = os.path.join(args.save_dir, 'avg_zstat_{}.nii'.format(name))
    nib.save(gd_stat_map_nii, stat_out_path)
    beta_out_path = os.path.join(args.save_dir, 'avg_beta_{}.nii'.format(name))
    nib.save(gd_beta_map_nii, beta_out_path)
