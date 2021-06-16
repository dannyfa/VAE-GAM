"""

This module contains smaller functions to be used in main model
module (vae_reg_GP.py) or throughout rest of code base.

"""
import torch
from torch import nn
import numpy as np
from scipy.stats import gamma
import pandas as pd

def hrf(times):
    """
    Args: time points for which we wish to estimate the HRF.
    Returns:
    Values for HRF at given time points.
    This is used to account for HRF when modeling biological/neural
    covariates -- e.g. task covariate in checker dset.
     """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    return values / np.max(values) * 0.6


def hrf_convolve(neural_response):
    """
    Convolves a neural response time series with the HRF.
    Args: neural response.
    This is the time series of values for the neural response variable
    we wish to convolve with the HRF.
    In this implementation, am assuming TR==1.4, which is the case for checker dset.
    But this can be easily made more flexible for other acquisition protocols/dsets.
    """
    if torch.is_tensor(neural_response):
        neural_response = neural_response.detach().cpu().numpy()
    tr_times = np.arange(0, 20, 1.4) #TR=1.4 here. Block lengths are 20s.
    hrf_at_trs = hrf(tr_times)
    convolved_series = np.convolve(neural_response, hrf_at_trs)
    n_to_remove = len(hrf_at_trs) - 1
    convolved_series = convolved_series[:-n_to_remove]
    return convolved_series

def get_xu_ranges(csv_file, eps = 1e-2):
    """
    Gets ranges for x values for GP inducing pts by rounding min/max values
    for each covariate across the entire dset.
    Args
    ----
    csv_file: file containing data. This is the same file passed to Data Class
    and loaders.
    """
    df = pd.read_csv(csv_file)
    mot_regrssors = ['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z']
    xu_ranges = []
    for reg in mot_regrssors:
        min, max = df[reg].min(), df[reg].max()
        xu_ranges.append([(min-eps), (max+eps)])
    return xu_ranges

def build_gp_params_dict(num_inducing_pts, device, csv_file):
    """
    Construct gp_params dict to hold linear weight + GP parameters
    (i.e., x's and y's for inducible points, kernel lengthscale and verticle variance)
    for each covariate.
    Args
    -----
    num_inducing_pts: Int.
    Number of inducing points to be used in each 1D GP.
    """
    gp_params  = {'task':{}, 'x':{}, 'y':{}, 'z':{}, \
    'xrot':{}, 'yrot':{}, 'zrot':{}}
    keys = list(gp_params.keys())
    subkeys = ['linW', 'xu', 'y', 'logkvar', 'log_ls']
    xu_ranges = get_xu_ranges(csv_file)
    for i in range(len(keys)):
            for j in subkeys:
                if j=='linW':
                    gp_params[keys[i]][j] = torch.nn.Parameter(torch.normal(mean = torch.tensor(0.0),\
                    std = torch.tensor(1.0)).to(device))
                if i!=0: #skip rest of gp params for task, which is binary variable
                    if j=='xu':
                        gp_params[keys[i]][j] = torch.linspace(xu_ranges[i-1][0], xu_ranges[i-1][1], \
                        num_inducing_pts).to(device)
                    elif j=='y':
                        gp_params[keys[i]][j] = torch.nn.Parameter(torch.rand(num_inducing_pts).to(device))
                    else:
                        gp_params[keys[i]][j] = torch.nn.Parameter(torch.as_tensor((0.0)).to(device))
    return gp_params

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
    binary sequence for control experiments involving large #3.
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
    '''
    Takes a df with samples, zscores each one of the motion regressor columns
    and replaces raw mot regressor inputs by their z-scored vals.
    Z-scoring is done for ALL vols and subjects at once in this case.
    '''
    mot_regrssors = df[['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z']]
    cols = list(mot_regrssors.columns)
    for col in cols:
        df[col] = (mot_regrssors[col] - mot_regrssors[col].mean())/mot_regrssors[col].std(ddof=0)
    return df
