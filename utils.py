"""

This module contains smaller functions to be used in main model
module (vae_reg_GP.py) or throughout rest of code base.

"""
import torch
from torch import nn
import numpy as np
from scipy.stats import gamma

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
    tr_times = np.arange(0, 20, 1.4) #TR=1.4 here. Block lengths are 20s.
    hrf_at_trs = hrf(tr_times)
    convolved_series = np.convolve(neural_response.detach().cpu().numpy(), hrf_at_trs)
    n_to_remove = len(hrf_at_trs) - 1
    convolved_series = convolved_series[:-n_to_remove]
    return torch.FloatTensor(convolved_series)

def build_gp_params_dict(num_inducing_pts, device):
    """
    Construct gp_params dict to hold linear weight + GP parameters
    (i.e., x's and y's for inducible points, kernel lengthscale and verticle variance)
    for each covariate.
    Args
    -----
    num_inducing_pts: Int.
    Number of inducing points to be used in each 1D GP.
    """
    keys = ['task', 'x', 'y', 'z', 'xrot', 'yrot', 'zrot']
    subkeys = ['linW', 'xu', 'y', 'logkvar', 'log_ls']
    xu_ranges = [[-4.00, 3.50], [-2.5, 3.42], [-3.45, 3.80], \
    [-3.31, 2.73], [-3.14, 4.76], [-3.03, 2.54]]
    gp_params  = {'task':{}, 'x':{}, 'y':{}, 'z':{}, \
    'xrot':{}, 'yrot':{}, 'zrot':{}}
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
