"""

This module contains smaller functions to be used in main model
module (vae_reg_GP.py) or throughout rest of code base.

"""

import torch
from torch import nn
import numpy as np
from scipy.stats import gamma
import pandas as pd
from copy import deepcopy


def hrf(times):
    """
    Args: time points for which we wish to estimate the HRF.
    Returns:
    Values for HRF at given time points.
    This is used to account for HRF when modeling biological/neural
    covariates -- e.g. visual stim covariate in checker dset.
     """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    return values / np.max(values) * 0.6


def get_xu_ranges(csv_files, eps = 1e-3):
    """
    Gets ranges for x values for GP inducing pts by rounding min/max values
    for each covariate across the entire dset.
    Args
    ----
    csv_file: file containing data. This is the same file passed to Data Class
    and loaders.
    """
    train_df = pd.read_csv(csv_files[0])
    test_df = pd.read_csv(csv_files[1])
    mot_regrssors = ['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z']
    xu_ranges = []
    for reg in mot_regrssors:
        min_val = min(train_df[reg].min(), test_df[reg].min())
        max_val = max(train_df[reg].max(), test_df[reg].max())
        xu_ranges.append([(min_val-eps), (max_val+eps)])
    return xu_ranges


def str2bool(v):
    """
    Str to Bool converter for wrapper script.
    This is used both for --from_ckpt and for --recons_only flags, which
    are False by default but can be turned on either by listing the flag (without args)
    or by listing with an appropriate arg (which can be converted to a corresponding boolean)
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def stimulus_to_neural(vol_times):
    """
    Creates binary sequence representing task variable in checker experiment/dataset.
    Each experimental block in this dset has 20s. First task block begins AFTER a block
    of NO-TASK.
    """
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

def control_stimulus_to_neural(vol_times):
    """
    Almost identical to stimulus_to_neural, except this is intended to create
    binary sequence for control experiments involving large 3.
    Here, the first stim block starts at time ==0. This was done so as to
    place artificial signal preferentially in volumes where no real V1 signal
    was present.
    """
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

def zscore(df):
    """
    Takes a df with samples, zscores each one of the motion regressor columns
    and replaces raw mot regressor inputs by their z-scored vals.
    Z-scoring is done for ALL vols and subjects at once in this case.
    """
    mot_regrssors = df[['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z']]
    cols = list(mot_regrssors.columns)
    for col in cols:
        df[col] = (mot_regrssors[col] - mot_regrssors[col].mean())/mot_regrssors[col].std(ddof=0)
    return df


def mk_spherical_mask(size, radius):
    """
    Creates spherical masks to be used inside add_control_signal script.
    Args
    -----
    size :: size of original 3D numpy matrix A.
    radius :: radius of sphere inside A which will be filled with ones.
    """
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
