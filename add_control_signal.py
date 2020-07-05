"""
Short script to add a control signal to original pre_processed data.

Control signals can be of 2 shapes:
1)4 small spheres added to frontal lobe.
2)A number -- in this case '3' added to frontal lobe.

These can be either convolved w/ HRF (as per usual) OR
Have a different (more challenging) link function.
Challenge link functions are of 2 types;
1) Linear w/ a saturation (linear_sat)
2) Inverted-V (inverted_delta)

Block-design chosen was opposite to one seen for real effect in V1 for all link functions AND shapes added.

DOES NOT overwrite original data -- instead, it writes output to same subdir as original.
Suffix 'altered', shape of signal, signal magnitude, link_function
and a time-stamp are added to output name to mark this operation.

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
import scipy
from scipy import ndimage #to rotate added signals
from copy import deepcopy
import torch
import torchvision
import torchvision.datasets as datasets # to get mnist digit dset
from PIL import Image

#get user args
parser = argparse.ArgumentParser(description='user args for adding control signal')

parser.add_argument('--root_dir', type=str, metavar='N', default='', \
help='Root dir where original .nii and .tsv files are located')
parser.add_argument('--intensity', type=float, metavar='N', default=600, \
help='Max abs value of signals added to data.')
parser.add_argument('--shape', type=str, metavar='N', default='simple', \
help='Shape of signal added. Simple refers to spheres. Any other str will yield a 7x7 hand-written 3.')
parser.add_argument('--radius', type=int, metavar='N', default=1, \
help='Radius of spheres to be added.Only used if type == simple')
parser.add_argument('--size', type=int, metavar='N', default=7, \
help='Dim of 3D array containing spherical masks. This is an A*A*A cube. Only used if type == simple')
parser.add_argument('--link_function', type=str, metavar='N', default='normal_hrf', \
help='Link function for added signal time series. Can be either normal_hrf, linear_sat or inverted_delta.')

args = parser.parse_args()

if args.root_dir=='':
    args.root_dir = os.getcwd()
else:
    if not os.path.exists(args.root_dir):
        print('Root dir given does not exist!')
        sys.exit()

#make sure link_function is one of 3 allowed options
if args.link_function not in ['normal_hrf', 'linear_sat', 'inverted_delta']:
    print('Link function given is NOT supported.')
    print('Please choose between normal_hrf, linear_sat OR inv_delta')
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

#creates a series with lenear increase up to 7th item
#f/b flattening (saturation)
def sat_link(times):
    out = []
    for t in times:
        if 0<=t<=6:
            res = (1/7)*(t)
            out.append(res)
        else:
            res = 1.0
            out.append(res)
    return out

#creates an inverted V-shaped time series
#assumes total of 14 pts in time series
#might need to be adjusted for other task block lengths
def inv_delta(times):
    out = []
    for t in times:
        if 0<=t<=6:
            res = (1/6)*(t)
            out.append(res)
        else:
            res = 1.0 - ((1/7)*(t-6))
            out.append(res)
    return(out)

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

if args.shape == 'simple':
    #create small sphere w/ desired intensity
    sphere = mk_spherical_mask(size=args.size, radius=args.radius)
    spherical_mask = args.intensity * sphere
    #create empty arr with same img dim to add control signal to
    control_sig = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
    #add 4 spheres to desired locations
    #these were chosen to be around frontal lobe by plotting/visual inspection
    #if size of sphere arr is changed, ranges have to be modified appropriately...
    #this can be made more fkexible in future iterations of code ...
    control_sig[15:22, 34:41, 14:21]+= spherical_mask
    control_sig[13:20, 38:45, 15:22]+= spherical_mask
    control_sig[20:27, 38:45, 15:22]+= spherical_mask
    control_sig[16:23, 38:45, 20:27]+= spherical_mask
else:
    #get mnist dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True,\
    transform=None)
    #create small list w/ 2-3 numbers we might use
    imgs = []
    for i, sample in enumerate(mnist_trainset):
        if i <=7:
            target = sample[1]
            #get a 2 and a 3. WIll use 3 only for now...
            if target == 2 or target ==3:
                img = sample[0]
                imgs.append(img)
            else:
                pass
    #get just one of these nmbers, say '3'
    three = imgs[1].resize((7, 7)) #resize it to 7x7.
    three = np.asarray(three)
    norm_three = three/255 #scale
    sig_three = args.intensity*norm_three #multiply by signal intensity
    rot_sig = ndimage.rotate(sig_three, -90) #this is needed given struct of fmri arr
    signal = np.broadcast_to(rot_sig, (10, 7, 7)) #broadcast to desired shape
    #create empty arr to hold control signal
    control_sig = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
    control_sig[15:25, 34:41, 9:16]+= signal

#now get time-series using link function
#TR and 0-20 range established based on acquisition & task design params for checker dset
if args.link_function == 'normal_hrf':
    TR=1.4
    vols = IMG_SHAPE[3]
    tr_times = np.arange(0, 20, TR)
    hrf_at_trs = hrf(tr_times)
    vol_times = np.arange(1, vols +1) * TR
    neural = stimulus_to_neural(vol_times)
    #convolve neural stim box-car series w/ HRF
    #take out last value to make arr lengths match
    time_series = np.convolve(neural, hrf_at_trs)
    n_to_remove = len(hrf_at_trs) - 1
    time_series = time_series[:-n_to_remove]
elif args.link_function == 'linear_sat':
    #build lin sat series for each task block
    task_times = np.arange(0, 14, 1)
    lin_sat_block = sat_link(task_times)
    #now build entire series w/ blocks of lin sat task effect
    #and blocks w/ out it
    #order of blocks is opposite of V1 effect in original
    time_series = np.zeros(98)
    time_series[0:14] += lin_sat_block
    time_series[28:42]+= lin_sat_block
    time_series[57:71]+= lin_sat_block
    time_series[83:97]+= lin_sat_block
else:
    #i.e., if link function is inverted_delta
    #build inverted delta series for each task block
    task_times = np.arange(0, 14, 1)
    inv_delta_block = inv_delta(task_times)
    #now build entire series w/ blocks w/ task effect
    #and blocks w/ out it
    #order of blocks is opposite of V1 effect in original
    time_series = np.zeros(98)
    time_series[0:14] += inv_delta_block
    time_series[28:42]+= inv_delta_block
    time_series[57:71]+= inv_delta_block
    time_series[83:97]+= inv_delta_block

#get date & intensity in str form
ts = datetime.datetime.now().date()
intensity_as_str = str(int(args.intensity))

#loop through subjs and create altered dataset
for i in range(len(subjs)):
    original_path = raw_data_files[i]
    orig_nii = nib.load(original_path) #each subj is its own ref for saving new set
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
    #save altered subj dataset to different path under same subdir
    #'_ALTERED_' suffix , control shape, signal intensity, link_function and date are added to output fname
    alt_path = original_path.rstrip('.nii.gz') +'_ALTERED_' + args.shape + '_' + intensity_as_str + \
    '_' + args.link_function + '_' + ts.strftime('%m_%d_%Y') + '.nii.gz'
    alt_nii = nib.Nifti1Image(altered_data, orig_nii.affine, orig_nii.header)
    nib.save(alt_nii, alt_path)
