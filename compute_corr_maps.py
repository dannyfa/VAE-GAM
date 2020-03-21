"""
Script to generate nearest-neighbors avg correlation maps.
Neighbors are taken as a 3D cross -- i.e., 2 in x, y and z respectively for each voxel
Correlation is Pearson correlation as implemented in scipy.stats

Does so for raw data and for model generated data (3 conditions: base, task and full_rec)
Maps come split into single subject versions and grand-avgs across subjects
Reference nii for each subj is taken from raw data for subj itself... For grandmaps,
a single subj raw map is chosen as reference. This is OK as long as subjs are all warped
into a common space.

Input Args:
:: raw_data_dir: Root dir where subj original (raw) data is stored under
:: model_recons_dir:: Root dir where model reconstructions are stored under
:: out_dir: Output dir where correlation maps are saved to. Outputs are split in
    2 folders, one for raw data and another for model generated data. Each will carry corresponding
    individual subj and avg maps.

TODOs
1. Make it obj oriented
2. Parallelize stuff for quicker compute time
3. Mk exception handling less repetitive...
4. Add Warning handling for pearsonr 
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

parser = argparse.ArgumentParser(description='user args for CorrMaps computation')

parser.add_argument('--raw_data_dir', type=str, metavar='N', default='', \
help='Root dir where raw subj nii files are stored  under.')
parser.add_argument('--model_recons_dir', type=str, metavar='N', default='', \
help='Root dir where model reconstructed volumes are stored  under.')
parser.add_argument('--out_dir', type=str, metavar='N', default='', \
help='Dir where output CorrMaps are saved to. Defauls to cwd if no arg is passed.')

args = parser.parse_args()

#setting up raw and model data_dirs
#add additional protention here!!
#mk using model gen data an option!!
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

#setting up sub-dirs for raw data and model recon Maps
raw_maps_dir = os.path.join(args.out_dir, 'raw_data_maps')
model_maps_dir = os.path.join(args.out_dir, 'model_data_maps')
if not os.path.exists(raw_maps_dir):
    os.makedirs(raw_maps_dir)
if not os.path.exists(model_maps_dir):
    os.makedirs(model_maps_dir)

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

#get concat time series of model reconstructions...
subj_recons = [] #list of dicts to handle recons for all subjs and conditions.
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

#Routine to build actual CorrMaps..
conditions = ['raw', 'base', 'task', 'full_rec']
for condition in conditions:
    #init grand_avg map
    grand_corr_map = np.zeros((41, 49, 35), np.float)
    #loop through subjs
    for s in range(len(subjs)):
        if condition == 'raw':
            data_path = raw_data_files[s]
            subj_nii = np.asarray(nib.load(data_path).dataobj)
        else:
            subj_nii = subj_recons[s][condition]
        x, y, z, times = subj_nii.shape[0], subj_nii.shape[1], \
        subj_nii.shape[2], subj_nii.shape[3]
        #get nearest-neighbors time series for each voxel
        nn = []
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    slcs = subj_nii[:, :, k, :] #get time series for kth slice
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
                        n5=subj_nii[i, j, k-1, :]
                    except:
                        n5=np.zeros(times)
                    try:
                        n6=subj_nii[i, j, k+1, :]
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
        #save subj CorrMap
        ref_nii = nib.load(raw_data_files[s]) #load subj raw file as ref
        if condition == 'raw':
            out_path = os.path.join(raw_maps_dir, '{}_CorrMap.nii'.format(subjs[s]))
            corr_map_nii = nib.Nifti1Image(corr_map, ref_nii.affine, ref_nii.header)
            nib.save(corr_map_nii, out_path)
        else:
            out_path = os.path.join(model_maps_dir, '{}_{}_CorrMap.nii'.format(subjs[s], condition))
            corr_map_nii = nib.Nifti1Image(corr_map, ref_nii.affine, ref_nii.header)
            nib.save(corr_map_nii, out_path)
        #add subj corr map to grand_avg
        grand_corr_map+=corr_map
    #calc grand avg after looping through subjs
    grand_corr_map = grand_corr_map/len(subjs) #compute avg across subjs
    #save grand_avg map
    ref_nii = nib.load(raw_data_files[0]) #get a nii file as ref. Can be any in this case ...
    grand_corr_map_nii = nib.Nifti1Image(grand_corr_map, ref_nii.affine, ref_nii.header)
    if condition =='raw':
        grand_out_path = os.path.join(raw_maps_dir, 'avg_CorrMap.nii')
    else:
        grand_out_path = os.path.join(model_maps_dir, '{}_avg_CorrMap.nii'.format(condition))
    nib.save(grand_corr_map_nii, grand_out_path)
