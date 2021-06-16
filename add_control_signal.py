"""
Short script to add a control signal to original pre_processed data.

Control signals can be of 2 shapes:
1)4 small spheres added to frontal lobe.
2)A number -- in this case as 13x13 hand-written '3' added to frontal lobe.
Number three is first binarized (mask of 1 or 0's) before being scaled and added
to controls.

These can be either: 1)simply multiplied by task block time-series (simple_ts) OR
3)convolved w/ HRF (immitating a biological signal).

DOES NOT overwrite original data -- instead, it writes output to same subdir as original.
Suffix 'ALTERED', shape of signal, signal magnitude, link_function
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
from copy import deepcopy
import torch
import torchvision
import torchvision.datasets as datasets
from PIL import Image

parser = argparse.ArgumentParser(description='User args for adding control signal.')

parser.add_argument('--root_dir', type=str, metavar='N', default='', \
help='Root dir where original .nii files are located')
parser.add_argument('--intensity', type=float, metavar='N', default=1000, \
help='Max abs value of signals added to data.')
parser.add_argument('--shape', type=str, metavar='N', default='Large3', \
help='Shape of signal added. Can also be simple, which corresponds to 4 small spheres placed in pre-specified locations in the FL.')
parser.add_argument('--radius', type=int, metavar='N', default=1, \
help='Radius of spheres to be added.Only used if shape == simple')
parser.add_argument('--size', type=int, metavar='N', default=7, \
help='Dim of 3D array containing spherical masks. This is an A*A*A cube. Only used if shape == simple')
parser.add_argument('--link_function', type=str, metavar='N', default='simple_ts', \
help='Link function for added signal time series. Can be either simple_ts or normal_hrf.')

args = parser.parse_args()

def _mk_spherical_mask(size, radius):
    '''
    Args:
    size :: size of original 3D numpy matrix A.
    radius :: radius of sphere inside A which will be filled with ones.
    '''
    s, r = size, radius
    #A : numpy.ndarray of shape size*size*size.
    A = np.zeros((size,size, size))
    #AA : copy of A
    AA = deepcopy(A)
    #(x0, y0, z0) : coordinates of center of circle inside A.
    x0, y0, z0 = int(np.floor(A.shape[0]/2)), \
    int(np.floor(A.shape[1]/2)), int(np.floor(A.shape[2]/2))

    for x in range(x0-radius, x0+radius+1):
        for y in range(y0-radius, y0+radius+1):
            for z in range(z0-radius, z0+radius+1):
                #deb: measures how far a coordinate in A is far from the center.
                #deb>=0: inside the sphere.
                #deb<0: outside the sphere.
                deb = radius - abs(x0-x) - abs(y0-y) - abs(z0-z)
                if (deb)>=0: AA[x,y,z] = 1
    return AA

if args.root_dir=='':
    args.root_dir = os.getcwd()
else:
    if not os.path.exists(args.root_dir):
        print('Root dir given does not exist!')
        sys.exit()

assert args.link_function in ['simple_ts', 'normal_hrf'], 'Link function NOT supported. Try either simple_ts or normal_hrf.'
assert args.shape in ['simple', 'Large3'], 'Shape type NOT supported! Try either simple or Large3.'

#get subjIDs
#excluded sub-A00058952 due to excess movement.
RE = re.compile('\Asub-A000*')
dirs = os.listdir(args.root_dir)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        if 'sub-A00058952' in dirs[i]:
            pass
        else:
            subjs.append(dirs[i])

#get paths to original pre-processed nii files
raw_data_files = []
for i in range(len(subjs)):
    full_path = os.path.join(args.root_dir, subjs[i])
    for data_file in Path(full_path).rglob('sub-A000*_preproc_bold_brainmasked_resampled.nii.gz'):
        raw_data_files.append(str(data_file))


IMG_SHAPE = (41, 49, 35, 98)

if args.shape == 'simple':
    #create small sphere w/ desired intensity
    sphere = _mk_spherical_mask(size=args.size, radius=args.radius)
    spherical_mask = args.intensity * sphere
    #create empty arr with same img dim to add control signal to
    control_sig = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
    #add 4 spheres to desired locations
    #these were chosen to be around frontal lobe by plotting/visual inspection
    #if size of sphere arr is changed, ranges have to be modified appropriately.
    control_sig[15:22, 34:41, 14:21]+= spherical_mask
    control_sig[13:20, 38:45, 15:22]+= spherical_mask
    control_sig[20:27, 38:45, 15:22]+= spherical_mask
    control_sig[16:23, 38:45, 20:27]+= spherical_mask
else:
    #will use large 3 signal.
    #get mnist dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True,\
    transform=None)
    imgs = []
    for i, sample in enumerate(mnist_trainset):
        if i <=10:
            target = sample[1]
            if target ==3 or target==0:
                img = sample[0]
                imgs.append(img)
            else:
                pass

    #create binary Large 3 signal
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
    #90 degrees rotation
    #needed given struct of fmri arr
    rot_sig = ndimage.rotate(sig, -90)
    #broadcast signal to desired shape
    signal = np.broadcast_to(rot_sig, (10, 13, 13))
    #create empty arr to hold control signal
    control_sig = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
    control_sig[15:25, 34:47, 9:22]+= signal

#now get time-series using link function
vols = IMG_SHAPE[3]
TR = 1.4
vol_times = np.arange(1, vols +1) * TR
neural = utils.control_stimulus_to_neural(vol_times)

if args.link_function == 'normal_hrf':
    time_series = utils.hrf_convolve(neural)
else:
    time_series = neural

#get date & intensity for filename extension
ts = datetime.datetime.now().date()
intensity_as_str = str(int(args.intensity))

#loop through subjs and create altered datasets
for i in range(len(subjs)):
    original_path = raw_data_files[i]
    orig_nii = nib.load(original_path)
    orig = np.array(orig_nii.dataobj)
    altered_data = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2], IMG_SHAPE[3]))
    #loop though vols
    #apply link function
    #and add to original
    for j in range(orig.shape[3]):
        vol = orig[:, :, :, j]
        added_signal = control_sig*time_series[j]
        vol += added_signal
        altered_data[:, :, :, j] = vol
    #save altered subj dataset to different filename under same subdir
    #'_ALTERED_' suffix , control shape, signal intensity, link_function and date are added to output fname
    alt_path = original_path.rstrip('.nii.gz') +'_ALTERED_' + args.shape + '_' + intensity_as_str + \
    '_' + args.link_function + '_' + ts.strftime('%m_%d_%Y') + '.nii.gz'
    alt_nii = nib.Nifti1Image(altered_data, orig_nii.affine, orig_nii.header)
    nib.save(alt_nii, alt_path)
