"""
Script for pre-processing data into useful samples for VAE-GP model
Writes output dset to csv file.
This file should be given as arg to FMRIDataset class.

Added:
HRF convolution piece -- makes task real-valued
Binary task category is still stored under task_bin var

Motion params -- x, y, z, pitch, roll and yaw.
These are per vol and per subj
If using original SPM files:
    these need to be resaved b/c scipy does not deal well with MatLab struct
If coming from fmriprep:
    just read tsv -- this is current version shown here!

Can be used for either checker set or for creating controls.
Main difference between these two is on StimToNeural function (see cmts below).

ToDO's:
- Mk it overall more flexible -- in terms of finding proper files & saving them with nice suffix
- Take away un-needed variables (& UTD DataClass script too)
"""

import os, sys
import re
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
import argparse
from sklearn import preprocessing # for norm step
import scipy.stats
from scipy.stats import gamma # for HRF funct
import scipy
from scipy import io # for loading motion mats

parser = argparse.ArgumentParser(description='user args for fMRIvae preproc')

parser.add_argument('--data_dir', type=str, metavar='N', default='', \
help='Root dir where nii files and .tsv file are located')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir where csv file to be given as input to FMRIDataset class is saved.')
parser.add_argument('--checker', type=bool, metavar='N', default=False, \
help='If true, will create csv file for actual checker task.')
parser.add_argument('--control', type=bool, metavar='N', default=False, \
help='If true, will create csv file for added signal control w/ specified link function.')
parser.add_argument('--link_function', type=str, metavar='N', default='normal_hrf', \
help='Link function for added (control) signal time series. IF creating checker csv, this MUST be normal hrf.')

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

#make sure that if checker_task == True, link_function == normal_hrf
if args.checker:
    assert args.link_function == 'normal_hrf', 'IF checker bool is True, link function MUST be normal_hrf!'

#make sure link_function is one of 3 allowed options
if args.link_function not in ['simple_ts', 'jittered_ts', 'normal_hrf', 'linear_sat', 'inverted_delta', 'inverted_u']:
    print('Link function given is NOT supported.')
    print('Please choose between simple_ts, jittered_ts, normal_hrf, linear_sat, inv_delta OR inverted_u')
    sys.exit()

#get subjIDs
#excluded sub-A00058952 due to high voxel intensity vals
#this subj had loads of movement!!! So will keep it out.
RE = re.compile('\Asub-A000*')
dirs = os.listdir(args.data_dir)
subjs = []
for i in range(len(dirs)):
    if RE.search(dirs[i]):
        if 'sub-A00058952' in dirs[i]:
            pass
        else:
            subjs.append(dirs[i])

# get paths to pre-processed nii files and to tsv mot files
# needs more flexibility in finding files w/ different names here...
raw_data_files = []
raw_reg_files = []
for i in range(len(subjs)):
    full_path = os.path.join(args.data_dir, subjs[i])
    for data_file in Path(full_path).rglob('sub-A000*_preproc_bold_brainmasked_resampled.nii.gz'):
        raw_data_files.append(str(data_file))
    for reg_file in Path(full_path).rglob('sub-A000*_ses-NFB2_task-CHECKERBOARD_acq-1400_desc-confounds_regressors.tsv'):
        raw_reg_files.append(str(reg_file))

#creating raw_df
raw_df = {'nii_files': raw_data_files, 'subjs': subjs, 'regressors': raw_reg_files}
raw_df = pd.DataFrame(raw_df)

#HRF funct.
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

#altered this to match intended block design for fake signals !!!
#this is opposite of task design for original V1 signal
def control_stimulus_to_neural(vol_times):
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

#this is the original stimulus to neural, used w/ checker set
def stimulus_to_neural(vol_times):
    t = vol_times//20
    res = []
    for i in t:
        if i==0:
            task=0
        elif i%2==0:
            task=0
        elif i%2!=0:
            task=1
        res.append(task)
    return(np.array(res))


#creates a series with linear increase up to 7th item
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

#creates an inverted_u series w/ 14 pts total
def inverted_u(x_coords):
    y_coords = [-x*x for x in x_coords]
    #make sure all #'s are + and max is 1 for given time range used here.
    y_coords = [ (x + 42.25)/42 for x in y_coords]
    return y_coords;

def zscore(df):
    '''
    Takes a df with samples, zscores each one of the motion regressor colums
    and replaces raw mot regressor inputs by their z-scored vals.
    Z-scoring is done for ALL vols and subjects at once in this case.
    Additionally, prints min and max for each z-scored column.
    This is useful for properly adjusting GP inducing pt x ranges...
    '''
    mot_regrssors = df[['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z']]
    cols = list(mot_regrssors.columns)
    print('Computing zcores and min/max for motion cols:')
    for col in cols:
        col_zscore = col + '_zscored'
        df[col] = (mot_regrssors[col] - mot_regrssors[col].mean())/mot_regrssors[col].std(ddof=0)
        min, max = df[col].min(), df[col].max()
        print('{}: min = {}; max = {}'.format(col_zscore, min, max))
    return df

#Building final csv file
samples = []
for i in raw_df['subjs']:
    #getting subjid, age, sex, path to raw nii files
    subjid = i
    #get motion params ...
    raw_reg = raw_df.loc[raw_df['subjs'] == i, 'regressors'].iloc[0]
    regressors = pd.read_csv(raw_reg, sep='\t', index_col=False)
    trans_x, trans_y, trans_z = regressors['trans_x'], regressors['trans_y'], regressors['trans_z']
    rot_x, rot_y, rot_z = regressors['rot_x'], regressors['rot_y'], regressors['rot_z']
    #now get fmri dset
    raw_nii = raw_df.loc[raw_df['subjs'] == i, 'nii_files'].iloc[0]
    fmri = np.array(nib.load(raw_nii).dataobj)
    #Get vol time series
    vols = fmri.shape[3]
    TR=1.4
    vol_times = np.arange(1, vols +1) * TR
    #use original stimToneural if checker_task==True
    if args.checker:
        neural = stimulus_to_neural(vol_times)
    else:
        neural = control_stimulus_to_neural(vol_times)

    if args.link_function == 'simple_ts':
        time_series = neural

    elif args.link_function == 'jittered_ts':
        #am setting both times_series and neural to jittered seq here
        times = np.arange(98)
        time_series = np.where((times%2 ==0), 1, 0)
        neural = time_series

    elif args.link_function == 'normal_hrf':
        tr_times = np.arange(0, 20, TR)
        hrf_at_trs = hrf(tr_times)
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

    elif args.link_function == 'inverted_delta':
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
    else:
        #get time pts
        inverted_u_times = np.arange(-6.5, 7.5, 1)
        #get corresponding inverted_u time time_series
        inverted_u_block = inverted_u(inverted_u_times)
        #now build entire series w/ blocks of inverted_u task effect
        #and blocks w/ out it
        #order of blocks is opposite of V1 effect in original
        time_series = np.zeros(98)
        time_series[0:14] += inverted_u_block
        time_series[28:42]+= inverted_u_block
        time_series[57:71]+= inverted_u_block
        time_series[83:97]+= inverted_u_block

    #finally, build samples...
    for vol in range(vols):
        sample = (subjid, vol, raw_nii, time_series[vol], neural[vol], trans_x[vol], \
        trans_y[vol], trans_z[vol], rot_x[vol], rot_y[vol], \
        rot_z[vol])
        samples.append(sample)

new_df = pd.DataFrame(list(samples), columns=["subjid","volume #", "nii_path", "task", \
"task_bin", "x", "y", "z", "rot_x", "rot_y", "rot_z"])
#z-score motion variables
zscored_df = zscore(new_df)
save_path = os.path.join(args.save_dir, 'preproc_dset_chckr_zscored_06012021.csv')
zscored_df.to_csv(save_path)
