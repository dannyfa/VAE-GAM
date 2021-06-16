"""
Short script to create avg SPM GLM stat map (for comparison) &
avg beta SPM map (for init in model).
Does so for checker effect (#1) and checker>fixation (#4).
"""

import os, sys
import re
from pathlib import Path
import nibabel as nib
import numpy as np
import argparse
from build_model_recons import _save_map

parser = argparse.ArgumentParser(description='args for calc SPM GLM avgs')

parser.add_argument('--data_dir', type=str, metavar='N', default='', \
help='Root dir where GLM beta and stat maps are.')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir to write out output avg maps.')
parser.add_argument('--ref_nii', type=str, metavar='N', default='', \
help='Ref nii from which hdr and affine are taken. Can be fmriprep pre-processed file from any subj.')

args = parser.parse_args()

def _get_maps(subjs, data_dir, map_type):
    """
    Creates a dict of dicts holding the different stat maps for each subject in subjs.
    Args
    ----
    subjs: list containing identifiers for subjects to be included.
    data_dir: root directory where results for GLM analyses for these subjects lives.
    Specific path here was hardcoded to represent set-up in our system/machine.
    However, user might wish to adapt it so that it matches their own system.
    map_type: str.
    Can be either 'stat' or 'beta', referring to either raw beta map output from GLM
    or maps resulting from computing stats on top of these maps.
    """
    if map_type == 'stat':
        file_pattern = 'thresh_zstat*.nii.gz'
    else:
        file_pattern = 'cope*.nii.gz'
    all_maps = {}
    for i in range(len(subjs)):
        subj_cons = {'cons1':{}, 'cons2':{}, 'cons3':{}, 'cons4':{}, 'cons5':{}}
        #path below was specific for set up on our machines. You might need to adjust it
        #so that it points to correct location in your system.
        full_path = os.path.join(data_dir, subjs[i],'ses-NFB2', 'func', \
        '{}_task-preproc_bold_brainmasked_resampled_corrected.feat'.format(subjs[i]))
        for f in Path(full_path).rglob(file_pattern):
            map_path = str(f)
            num = re.findall(r'\d+', map_path)[-1]
            map = np.array(nib.load(map_path).dataobj)
            key = 'cons{}'.format(num)
            try:
                subj_cons[key] = map
            except:
                pass
        all_maps[subjs[i]] = subj_cons
    return all_maps

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

#get subjIDs
#excluded sub-A00058952 due to high voxel intensity vals
RE = re.compile('\Asub-A000*')
dirs = os.listdir(args.data_dir)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        if 'sub-A00058952' in dirs[i]:
            pass
        else:
            subjs.append(dirs[i])

#get needed stat and beta maps
stat_maps = _get_maps(subjs, args.data_dir, 'stat')
beta_maps = _get_maps(subjs, args.data_dir, 'beta')

#now compute avg_maps
for i in range (1,6): #range chosen based on number of contrasts, conditions for chcker GLM analysis.
    gd_stat_map = np.zeros((41, 49, 35), dtype=np.float32)
    gd_beta_map = np.zeros((41, 49, 35), dtype=np.float32)
    name = 'cons{}'.format(i)
    for subj in subjs:
        gd_stat_map += stat_maps[subj][name]
        gd_beta_map += beta_maps[subj][name]
    #div though len subjs...
    gd_stat_map = gd_stat_map/len(subjs)
    gd_beta_map = gd_beta_map/len(subjs)
    #save maps
    _save_map(gd_stat_map, args.ref_nii, args.save_dir, 'zstat_{}'.format(name))
    _save_map(gd_beta_map, args.ref_nii, args.save_dir, 'beta_{}'.format(name))
