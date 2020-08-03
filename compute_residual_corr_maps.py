"""
Script to generate VAE residuals correlation maps for NN voxels.
VAE residuals are calculated (per subj) as follows:
    res = original task vol - corresponding (model predicted) task volume
This is done for all volumes labeled as containing task signal (i.e., where task_bin ==1), for each subj.
Neighbors are taken as a 3D cross -- i.e., 2 in x, y and z respectively for each voxel
Correlation is Pearson correlation as implemented in scipy.stats

Maps come split into single subject versions and grand-avg across subjects
Reference nii for each subj is taken from raw data for subj itself... For grandmaps,
a single subj raw map is chosen as reference. This is OK as long as subjs are all warped
into a common (in this case MNI) space.

Input Args:
:: raw_data_dir: Root dir where subj original (raw) data is stored under
:: model_recons_dir:: Root dir where model reconstructions are stored under
:: out_dir: Output dir where correlation maps are saved to.

TODOs
1. Make it obj oriented
2. Parallelize stuff for quicker compute time
3. Mk exception handling less repetitive in NN search routine
4. Handle pearsonr warning better
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr
import os, sys
import re
from pathlib import Path
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='user args for ResCorrMaps computation')

parser.add_argument('--raw_data_dir', type=str, metavar='N', default='', \
help='Root dir where raw subj nii files are stored  under.')
parser.add_argument('--model_recons_dir', type=str, metavar='N', default='', \
help='Root dir where model reconstructed volumes are stored  under.')
parser.add_argument('--csv_file', type=str, metavar='N', default='', \
help='csv file for dset. This is the same generated by the preprocessing script.')
parser.add_argument('--out_dir', type=str, metavar='N', default='', \
help='Dir where output CorrMaps are saved to. Defauls to cwd if no arg is passed.')

args = parser.parse_args()

if args.raw_data_dir=='':
    args.raw_data_dir = os.getcwd()
else:
    if not os.path.exists(args.raw_data_dir):
        print('Raw Data dir given does not exist!')
        print('Cannot proceed w/out raw subj data!')
        sys.exit()

if args.model_recons_dir=='':
    args.model_recons_dir = os.getcwd()
else:
    if not os.path.exists(args.model_recons_dir):
        print('Model recons dir given does not exist!')
        print('Cannot proceede w/out model generated data!')
        sys.exit()

#setting up out_dir
if args.out_dir == '':
    args.out_dir = os.getcwd()
else:
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    else:
        pass

#get subjIDs
#excluded sub-A00058952 due to high voxel intensity vals...
RE = re.compile('\Asub-A000*')
dirs = os.listdir(args.raw_data_dir)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        if 'sub-A00058952' in dirs[i]:
            pass
        else:
            subjs.append(dirs[i])

#get paths to original pre-processed nii files for subjs...
raw_data_files = []
for i in range(len(subjs)):
    full_path = os.path.join(args.raw_data_dir, subjs[i])
    for data_file in Path(full_path).rglob('sub-A000*_preproc_bold_brainmasked_resampled.nii.gz'):
        raw_data_files.append(str(data_file))

#get concatenated volumes time-series of model reconstructions...
#create list of dicts to handle recons for all subjs and conditions.
subj_recons = []
for subj in subjs:
    maps = {'base':{}, 'task':{}, 'full_rec':{}}
    data_root = os.path.join(args.model_recons_dir, subj)
    vol_dirs = os.listdir(data_root)
    #sort these out by vol#
    #this is necessary to recon (x,y,z, time) arr faithfully
    sorted_vols = sorted(vol_dirs, key= lambda k:(int(re.findall(r'\d+', k)[0])))
    for i in list(maps.keys()):
        map = []
        for vol_dir in sorted_vols:
            #get paths and vols
            path = os.path.join(data_root, vol_dir, 'recon_{}.nii'.format(i))
            vol = np.array(nib.load(path).dataobj)
            map.append(vol)
        maps[i] = np.moveaxis(np.asarray(map), 0, -1)
    subj_recons.append(maps)

#get csv w/ dset
dset = pd.read_csv(args.csv_file)
#get idxs for all task vols
#these are the same for all subjs (d/t block design)
#so ok to get it from any sample subj, say subj 57808
is_subj = dset['subjid'] == 'sub-A00057808'
subj_57808 = dset[is_subj]
is_task = subj_57808['task_bin'] == 1
subj_57808_task = subj_57808[is_task]
task_idxs = subj_57808_task['volume #']

#routine to calc residual volumes and aggregate their time series, for each subj
#list to handle residuals time-series for all subjs
res_time_series = []
#loop through subjs to get residual vol time-series for each subj
for s in range(len(subjs)):
    orig_map = np.array(nib.load(raw_data_files[s]).dataobj)
    model_map = subj_recons[s]['task']
    subj_res_series = []
    #loop though volumes
    for t in range(orig_map.shape[3]):
        #check if vol is a task volume...
        if t in task_idxs.values:
            #if yes, calc res for that volume
            orig_vol = orig_map[:, :, :, t]
            model_vol = model_map[:, :, :, t]
            res = orig_vol - model_vol
            #and append it to subj's residual time-series
            subj_res_series.append(res)
        else:
            pass
    subj_res_series = np.moveaxis(np.asarray(subj_res_series), 0, -1)
    res_time_series.append(subj_res_series)

#now compute single subj and grand avg ResCorr Maps
#init grand_avg map
grand_rescorr_map = np.zeros((41, 49, 35), np.float)
#loop through subjs
for s in range(len(subjs)):
    subj_time_series = res_time_series[s]
    x, y, z, times = subj_time_series.shape[0], subj_time_series.shape[1], \
    subj_time_series.shape[2], subj_time_series.shape[3]
    #get nearest-neighbors time series for each voxel
    nn = []
    for i in range(x):
        for j in range(y):
            for k in range(z):
                slcs = subj_time_series[:, :, k, :] #get time series for kth axial slice
                center = slcs[i, j, :] #get series for cross center
                #get neighbors
                #exceptions catch edges for which idx would be out of range
                #if a given nn is out of range for arr, its series is made to be simply zeroes
                #nn in y
                try:
                    n1=slcs[i, j-1, :]
                except:
                    n1 = np.zeros(times)
                try:
                    n2=slcs[i, j+1, :]
                except:
                    n2=np.zeros(times)
                #nn in x
                try:
                    n3=slcs[i-1, j, :]
                except:
                    n3=np.zeros(times)
                try:
                    n4=slcs[i+1, j, :]
                except:
                    n4 = np.zeros(times)
                #nn in z
                try:
                    n5=subj_time_series[i, j, k-1, :]
                except:
                    n5=np.zeros(times)
                try:
                    n6=subj_time_series[i, j, k+1, :]
                except:
                    n6=np.zeros(times)
                neighbors = np.stack((center, n1, n2, n3, n4, n5, n6), axis=0)
                nn.append(neighbors)
    nn = np.asarray(nn)
    #now calc CorrMap
    corr_map = np.zeros(nn.shape[0])
    for voxel in range(nn.shape[0]):
        #calc correlations
        voxel_set = nn[voxel, :, :]
        correlations = []
        center_series = voxel_set[0, :]
        for n in range(1, voxel_set.shape[0]):
            corr, _ = pearsonr(center_series, voxel_set[n, :])
            if corr == 'nan':
                corrected_corr = 0 #if pearsonr returns nan for cte input array, just make corr=0
            else:
                corrected_corr = corr
            correlations.append(corrected_corr)
        avg_corr = np.asarray(correlations).mean()
        corr_map[voxel]= avg_corr
    corr_map = corr_map.reshape(x, y, z)
    #save subj ResCorrMap
    ref_nii = nib.load(raw_data_files[s]) #load subj raw file as ref
    out_path = os.path.join(args.out_dir, '{}_ResCorrMap.nii'.format(subjs[s]))
    corr_map_nii = nib.Nifti1Image(corr_map, ref_nii.affine, ref_nii.header)
    nib.save(corr_map_nii, out_path)
    #add subj corr map to grand_avg
    grand_rescorr_map+=corr_map

#calc grand avg after looping through subjs
grand_rescorr_map = grand_rescorr_map/len(subjs) #compute avg across subjs
#save grand_avg map
ref_nii = nib.load(raw_data_files[0]) #get a nii file as ref. Can be any in this case ...
grand_rescorr_map_nii = nib.Nifti1Image(grand_rescorr_map, ref_nii.affine, ref_nii.header)
grand_out_path = os.path.join(args.out_dir, 'grand_avg_ResCorrMap.nii')
nib.save(grand_rescorr_map_nii, grand_out_path)