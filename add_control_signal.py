"""
Short script to add a control signal to original pre_processed data.

Control signal consists of 4 small spheres added to frontal lobe.

DOES NOT overwrite original data -- instead, it writes output to same subdir as original.
Suffix 'altered', signal magnitude and a time-stamp are added to output name to mark this operation.

Artifitial signal time series is first convolved with HRF prior to adding to volume time series
Block-design chosen was opposite to one seen for real effect in V1.

"""
import os, sys
import re
import datetime
from pathlib import Path
import nibabel as nib
import numpy as np
import argparse
import scipy.stats
from scipy.stats import gamma # for HRF funct
from copy import deepcopy

#get user args
parser = argparse.ArgumentParser(description='user args for adding control signal')

parser.add_argument('--root_dir', type=str, metavar='N', default='', \
help='Root dir where original .nii and .tsv files are located')
parser.add_argument('--intensity', type=float, metavar='N', default=500, \
help='Abs value of spherical signals added to data.')
parser.add_argument('--radius', type=int, metavar='N', default=1, \
help='Radius of spheres to be added.')
parser.add_argument('--size', type=int, metavar='N', default=7, \
help='Size of 3D array containing spherical masks. This is a cube of dim A*A*A')

args = parser.parse_args()

if args.root_dir=='':
    args.root_dir = os.getcwd()
else:
    if not os.path.exists(args.root_dir):
        print('Root dir given does not exist!')
        sys.exit()

#define helper functions

def mk_spherical_mask(size, radius):
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

def hrf(times):
    """ Return values for HRF at given times """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

#this is very similar to stim_to_neural function used for actual task effect
#except for flipped block design
#i.e., only blocks with NO V1 effect will get control signal
def stimulus_to_neural(vol_times):
    t = vol_times//20
    res = []
    for i in t:
        if i==0:
            task=1
        elif i%2==0:
            task=1
        elif i%2!=0:
            task=0
        res.append(task)
    return(np.array(res))

#get subjIDs
#excluded sub-A00058952 due to high voxel intensity vals
#this subj had loads of movement!!!
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

#these are dims for checker dset
IMG_SHAPE = (41, 49, 35, 98)
#create small sphere w/ desired intensity
sphere = mk_spherical_mask(size=args.size, radius=args.radius)
spherical_mask = args.intensity * sphere
#create empty arr with same img dim to add control signal to
control_sig = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
#add 4 spheres to desired locations
#these were chosen to be around frontal lobe by plotting/visual inspection
#if size of sphere arr is changed, ranges have to be modified appropriately...
#this can be made more fkexible if desired ...
control_sig[15:22, 34:41, 14:21]+= spherical_mask
control_sig[13:20, 38:45, 15:22]+= spherical_mask
control_sig[20:27, 38:45, 15:22]+= spherical_mask
control_sig[16:23, 38:45, 20:27]+= spherical_mask

#now get time series convolution
#TR and 0-20 range established based on acquisition & task design params for checker dset
TR=1.4
vols = IMG_SHAPE[3]
tr_times = np.arange(0, 20, TR)
hrf_at_trs = hrf(tr_times)
vol_times = np.arange(1, vols +1) * TR
neural = stimulus_to_neural(vol_times)
#convolve neural stim box-car series w/ HRF
#take out last value to make arr lengths match
convolved = np.convolve(neural, hrf_at_trs)
n_to_remove = len(hrf_at_trs) - 1
convolved = convolved[:-n_to_remove]

#get date
ts = datetime.datetime.now().date()
intensity_as_str = str(int(args.intensity))
#loop through subjs and create altered dataset
for i in range(len(subjs)):
    original_path = raw_data_files[i]
    orig_nii = nib.load(original_path) #each subj is its own ref for saving new set
    orig = np.array(orig_nii.dataobj)
    altered_data = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2], IMG_SHAPE[3]))
    #loop though vols
    #convolve them with hrf
    #and add to original
    for j in range(orig.shape[3]):
        vol = orig[:, :, :, j]
        conv_signal = control_sig*convolved[j]
        vol += conv_signal
        altered_data[:, :, :, j] = vol
    #save alt subj dataset to diff path under same subdir
    #'_ALTERED_' suffix , signal intensity and date are added to output fname
    alt_path = original_path.rstrip('.nii.gz') +'_ALTERED_' + intensity_as_str + \
    '_' + ts.strftime('%m_%d_%Y') + '.nii.gz'
    alt_nii = nib.Nifti1Image(altered_data, orig_nii.affine, orig_nii.header)
    nib.save(alt_nii, alt_path)
