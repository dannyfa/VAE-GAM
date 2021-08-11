"""
Short script to add a control signal to original pre_processed nifti data.

Control signals can be of 2 shapes:
1)4 small spheres added to frontal lobe. (NOT used in actual paper, but do work.)
2)A number -- in this case as 13x13 hand-written '3' added to frontal lobe.
Number three is first binarized (mask of 1 or 0's) before being scaled and added
to existing nifti data.

DOES NOT overwrite original data -- instead, it writes output to same subdir as original.
Suffix 'ALTERED', shape of signal, signal magnitude/scale, 'simple_ts'
and a date-stamp are added to output name to mark this operation.

"""
import utils
import os, sys
import re
import datetime
from pathlib import Path
import nibabel as nib
import numpy as np
import argparse
import scipy
from scipy import ndimage
import torch
import torchvision
import torchvision.datasets as datasets
from PIL import Image

parser = argparse.ArgumentParser(description='user args for add_control_signal script.')

parser.add_argument('--root_dir', type=str, metavar='N', default='', \
help='Root dir where original .nii and .tsv files are located.')
parser.add_argument('--intensity', type=float, metavar='N', default=1000, \
help='Intensity of synthetic signal added to data.')
parser.add_argument('--shape', type=str, metavar='N', default='simple', \
help='Shape of signal added. Simple refers to 4 spheres. Any other str will yield a hand-written 3.')
parser.add_argument('--radius', type=int, metavar='N', default=1, \
help='Radius of spheres to be added. Only used if shape == simple.')
parser.add_argument('--size', type=int, metavar='N', default=7, \
help='Dim of 3D array containing spherical masks. This is an A*A*A cube. Only used if shape == simple')
parser.add_argument('--nii_file_pattern', type=str, metavar='N', default='sub-A000*_preproc_bold_brainmasked_resampled.nii.gz', \
help='General pattern for filenames of nifti files to be used. Can contain any wildcard that glob and rgob can handle.')

args = parser.parse_args()

if args.root_dir=='':
    args.root_dir = os.getcwd()
else:
    if not os.path.exists(args.root_dir):
        print('Root dir given does not exist!')
        sys.exit()

#get subjIDs
#excluded sub-A00058952 d/t excess movement.
RE = re.compile('\Asub-A000*')
dirs = os.listdir(args.root_dir)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        if 'sub-A00058952' in dirs[i]:
            pass
        else:
            subjs.append(dirs[i])

#get paths to original pre-processed .nii files
raw_data_files = []
for i in range(len(subjs)):
    full_path = os.path.join(args.root_dir, subjs[i])
    for data_file in Path(full_path).rglob(args.nii_file_pattern):
        raw_data_files.append(str(data_file))

IMG_SHAPE = (41, 49, 35, 98)

if args.shape == 'simple':
    #create small sphere w/ desired intensity
    sphere = utils.mk_spherical_mask(size=args.size, radius=args.radius)
    spherical_mask = args.intensity * sphere
    #create empty arr with same img dim to add control signal to
    control_sig = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
    #add 4 spheres to desired locations
    #these were chosen to be around frontal lobe
    #if size of sphere is changed, ranges used will have to be modified appropriately...
    control_sig[15:22, 34:41, 14:21]+= spherical_mask
    control_sig[13:20, 38:45, 15:22]+= spherical_mask
    control_sig[20:27, 38:45, 15:22]+= spherical_mask
    control_sig[16:23, 38:45, 20:27]+= spherical_mask
else:
    #will add large (13x13) binary 3 synthetic signal.
    #get mnist dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True,\
    transform=None)
    imgs = []
    for i, sample in enumerate(mnist_trainset):
        if i <=10:
            target = sample[1]
            #get all 0s OR 3s.
            #In practice, I ended up using number '3' only.
            if target == 0 or target ==3:
                img = sample[0]
                imgs.append(img)
            else:
                pass

    #create binary 13x13 '3' signal
    three = imgs[1].resize((13, 13))
    three = np.asarray(three)
    norm_three = three/255
    sig_mean = np.mean(norm_three.flatten())
    sig_std = np.std(norm_three.flatten())
    #am setting all pixels above 0.85std to 1 and rest to 0.
    binary_sig = (np.where(norm_three.flatten() > (sig_mean + 0.85*sig_std), 1, 0)).reshape(norm_three.shape[0], \
    norm_three.shape[1])
    #scale by signal intensity
    sig = args.intensity*binary_sig
    #rotate by 90 degrees. THis puts it in correct orientation w.r.t. fMRI nifti coordinates.
    rot_sig = ndimage.rotate(sig, -90)
    #broadcast signal to desired shape
    signal = np.broadcast_to(rot_sig, (10, 13, 13))
    #create empty arr to hold control signal
    control_sig = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
    #and add synthetic signal to this empty array.
    control_sig[15:25, 34:47, 9:22]+= signal

#create control stimulus time series
#TR, #vols used is the same as original data to which synthetic signal is being added.
vols = IMG_SHAPE[3]
TR = 1.4
vol_times = np.arange(1, vols +1) * TR
neural = utils.control_stimulus_to_neural(vol_times)

#put date & intensity in str form
ts = datetime.datetime.now().date()
intensity_as_str = str(int(args.intensity))

#loop through subjs and create altered dataset
for i in range(len(subjs)):
    original_path = raw_data_files[i]
    orig_nii = nib.load(original_path)
    orig = np.array(orig_nii.dataobj)
    altered_data = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2], IMG_SHAPE[3]))
    #loop though vols
    #and add synthetic signal
    for j in range(orig.shape[3]):
        vol = orig[:, :, :, j]
        added_signal = control_sig*neural[j]
        vol += added_signal
        altered_data[:, :, :, j] = vol
    #save altered subj dataset to different path under same subdir
    #'_ALTERED_' suffix , control shape, signal intensity, 'simple_ts' and date are added to output fname
    alt_path = original_path.rstrip('.nii.gz') +'_ALTERED_' + args.shape + '_' + intensity_as_str + \
    '_simple_ts_' + ts.strftime('%m_%d_%Y') + '.nii.gz'
    alt_nii = nib.Nifti1Image(altered_data, orig_nii.affine, orig_nii.header)
    nib.save(alt_nii, alt_path)
