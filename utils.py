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
import re
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.tensorboard import SummaryWriter
from scipy import ndimage
from scipy.stats import norm
import os
import subprocess
import nibabel as nib

#########
#Miscellaneous Helpers
#########

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


######################
#methods to log maps, GP params and etc during training
######################

def log_qu_plots(epoch, gp_params, writer, log_type):
    """
    Creates q(u) plots which can be passed as figs to TB.
    Should be called after each epoch uptade.
    """
    #get means (qu_m), covariance mat (qu_S) and xu ranges for each covariate
    #x
    qu_m_x = gp_params['x']['qu_m'].detach().cpu().numpy().reshape(6)
    qu_S_x = np.diag(gp_params['x']['qu_S'].detach().cpu().numpy())
    xu_x = gp_params['x']['xu'].detach().cpu().numpy()
    #y
    qu_m_y = gp_params['y']['qu_m'].detach().cpu().numpy().reshape(6)
    qu_S_y = np.diag(gp_params['y']['qu_S'].detach().cpu().numpy())
    xu_y = gp_params['y']['xu'].detach().cpu().numpy()
    #z
    qu_m_z = gp_params['z']['qu_m'].detach().cpu().numpy().reshape(6)
    qu_S_z = np.diag(gp_params['z']['qu_S'].detach().cpu().numpy())
    xu_z = gp_params['z']['xu'].detach().cpu().numpy()
    #xrot
    qu_m_xrot = gp_params['xrot']['qu_m'].detach().cpu().numpy().reshape(6)
    qu_S_xrot = np.diag(gp_params['xrot']['qu_S'].detach().cpu().numpy())
    xu_xrot = gp_params['xrot']['xu'].detach().cpu().numpy()
    #yrot
    qu_m_yrot = gp_params['yrot']['qu_m'].detach().cpu().numpy().reshape(6)
    qu_S_yrot = np.diag(gp_params['yrot']['qu_S'].detach().cpu().numpy())
    xu_yrot = gp_params['yrot']['xu'].detach().cpu().numpy()
    #zrot
    qu_m_zrot = gp_params['zrot']['qu_m'].detach().cpu().numpy().reshape(6)
    qu_S_zrot = np.diag(gp_params['zrot']['qu_S'].detach().cpu().numpy())
    xu_zrot = gp_params['zrot']['xu'].detach().cpu().numpy()

    #now create figure
    fig, axs = plt.subplots(3,2, figsize=(15, 15))
    axs[0,0].plot(xu_x, qu_m_x, c='darkblue', alpha=0.5, label = 'q(u) posterior mean')
    x_two_sigma = 2*np.sqrt(qu_S_x)
    kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
    axs[0,0].fill_between(xu_x, (qu_m_x-x_two_sigma), (qu_m_x+x_two_sigma), **kwargs)
    axs[0,0].legend(loc='best')
    axs[0,0].set_title('q(u) x covariate at epoch {}'.format(epoch))
    axs[0,0].set_xlabel('Covariate x -- x vals ')
    axs[0,0].set_ylabel('q(u)')

    axs[0,1].plot(xu_y, qu_m_y, c='darkblue', alpha=0.5, label = 'q(u) posterior mean')
    y_two_sigma = 2*np.sqrt(qu_S_y)
    kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
    axs[0,1].fill_between(xu_y, (qu_m_y-y_two_sigma), (qu_m_y+y_two_sigma), **kwargs)
    axs[0,1].legend(loc='best')
    axs[0,1].set_title('q(u) y covariate at epoch {}'.format(epoch))
    axs[0,1].set_xlabel('Covariate y -- x vals ')
    axs[0,1].set_ylabel('q(u)')

    axs[1,0].plot(xu_z, qu_m_z, c='darkblue', alpha=0.5, label = 'q(u) posterior mean')
    z_two_sigma = 2*np.sqrt(qu_S_z)
    kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
    axs[1,0].fill_between(xu_z, (qu_m_z-z_two_sigma), (qu_m_z+z_two_sigma), **kwargs)
    axs[1,0].legend(loc='best')
    axs[1,0].set_title('q(u) z covariate at epoch {}'.format(epoch))
    axs[1,0].set_xlabel('Covariate z -- x vals ')
    axs[1,0].set_ylabel('q(u)')

    axs[1,1].plot(xu_xrot, qu_m_xrot, c='darkblue', alpha=0.5, label = 'q(u) posterior mean')
    xrot_two_sigma = 2*np.sqrt(qu_S_xrot)
    kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
    axs[1,1].fill_between(xu_xrot, (qu_m_xrot-xrot_two_sigma), (qu_m_xrot+xrot_two_sigma), **kwargs)
    axs[1,1].legend(loc='best')
    axs[1,1].set_title('q(u) xrot covariate at epoch {}'.format(epoch))
    axs[1,1].set_xlabel('Covariate xrot -- x vals ')
    axs[1,1].set_ylabel('q(u)')

    axs[2,0].plot(xu_yrot, qu_m_yrot, c='darkblue', alpha=0.5, label = 'q(u) posterior mean')
    yrot_two_sigma = 2*np.sqrt(qu_S_yrot)
    kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
    axs[2,0].fill_between(xu_yrot, (qu_m_yrot-yrot_two_sigma), (qu_m_yrot+yrot_two_sigma), **kwargs)
    axs[2,0].legend(loc='best')
    axs[2,0].set_title('q(u) yrot covariate at epoch {}'.format(epoch))
    axs[2,0].set_xlabel('Covariate yrot -- x vals ')
    axs[2,0].set_ylabel('q(u)')

    axs[2,1].plot(xu_zrot, qu_m_zrot, c='darkblue', alpha=0.5, label = 'q(u) posterior mean')
    zrot_two_sigma = 2*np.sqrt(qu_S_zrot)
    kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
    axs[2,1].fill_between(xu_zrot, (qu_m_zrot-zrot_two_sigma), (qu_m_zrot+zrot_two_sigma), **kwargs)
    axs[2,1].legend(loc='best')
    axs[2,1].set_title('q(u) zrot covariate at epoch {}'.format(epoch))
    axs[2,1].set_xlabel('Covariate zrot -- x vals ')
    axs[2,1].set_ylabel('q(u)')

    #and pass it to TB writer
    writer.add_figure("q(u)_{}".format(log_type), fig)

def log_qkappa_plots(gp_params, writer, log_type):
    """
    Logs q(k) to tensorboard.
    Plots only posterior --> prior is N(1, 0.5^2).
    """
    #task
    sa_task = gp_params['task']['sa'].detach().cpu().numpy().reshape(1)
    std_task = np.exp(gp_params['task']['logstd'].detach().cpu().numpy())
    task_gauss = norm(sa_task[0], scale = std_task[0])
    x_task = np.linspace(task_gauss.ppf(0.01), task_gauss.ppf(0.99), 100)
    y_task = task_gauss.pdf(x_task)
    #x
    sa_x= gp_params['x']['sa'].detach().cpu().numpy().reshape(1)
    std_x = np.exp(gp_params['x']['logstd'].detach().cpu().numpy())
    x_gauss = norm(sa_x[0], scale = std_x[0])
    x_x = np.linspace(x_gauss.ppf(0.01), x_gauss.ppf(0.99), 100)
    y_x = x_gauss.pdf(x_x)
    #y
    sa_y= gp_params['y']['sa'].detach().cpu().numpy().reshape(1)
    std_y = np.exp(gp_params['y']['logstd'].detach().cpu().numpy())
    y_gauss = norm(sa_y[0], scale = std_y[0])
    x_y = np.linspace(y_gauss.ppf(0.01), y_gauss.ppf(0.99), 100)
    y_y = y_gauss.pdf(x_y)
    #z
    sa_z= gp_params['z']['sa'].detach().cpu().numpy().reshape(1)
    std_z = np.exp(gp_params['z']['logstd'].detach().cpu().numpy())
    z_gauss = norm(sa_z[0], scale = std_z[0])
    x_z = np.linspace(z_gauss.ppf(0.01), z_gauss.ppf(0.99), 100)
    y_z = z_gauss.pdf(x_z)
    #xrot
    sa_xrot= gp_params['xrot']['sa'].detach().cpu().numpy().reshape(1)
    std_xrot = np.exp(gp_params['xrot']['logstd'].detach().cpu().numpy())
    xrot_gauss = norm(sa_xrot[0], scale = std_xrot[0])
    x_xrot = np.linspace(xrot_gauss.ppf(0.01), xrot_gauss.ppf(0.99), 100)
    y_xrot = xrot_gauss.pdf(x_xrot)
    #yrot
    sa_yrot= gp_params['yrot']['sa'].detach().cpu().numpy().reshape(1)
    std_yrot = np.exp(gp_params['yrot']['logstd'].detach().cpu().numpy())
    yrot_gauss = norm(sa_yrot[0], scale = std_yrot[0])
    x_yrot = np.linspace(yrot_gauss.ppf(0.01), yrot_gauss.ppf(0.99), 100)
    y_yrot = yrot_gauss.pdf(x_yrot)
    #zrot
    sa_zrot= gp_params['zrot']['sa'].detach().cpu().numpy().reshape(1)
    std_zrot = np.exp(gp_params['zrot']['logstd'].detach().cpu().numpy())
    zrot_gauss = norm(sa_zrot[0], scale = std_zrot[0])
    x_zrot = np.linspace(zrot_gauss.ppf(0.01), zrot_gauss.ppf(0.99), 100)
    y_zrot = zrot_gauss.pdf(x_zrot)
    #sex
    sa_sex = gp_params['sex']['sa'].detach().cpu().numpy().reshape(1)
    std_sex = np.exp(gp_params['sex']['logstd'].detach().cpu().numpy())
    sex_gauss = norm(sa_sex[0], scale = std_sex[0])
    x_sex = np.linspace(sex_gauss.ppf(0.01), sex_gauss.ppf(0.99), 100)
    y_sex = sex_gauss.pdf(x_sex)

    #now create plot
    fig, axs = plt.subplots(3,3, figsize=(15, 15))
    axs[0,0].plot(x_task, y_task, lw=2, alpha = 0.5, color = 'green')
    axs[0,0].set_title('Task q(k)')
    axs[0,1].plot(x_x, y_x, lw=2, alpha = 0.5, color = 'blue')
    axs[0,1].set_title('X q(k)')
    axs[0,2].plot(x_y, y_y, lw=2, alpha = 0.5, color = 'orange')
    axs[0,2].set_title('Y q(k)')
    axs[1,0].plot(x_z, y_z, lw=2, alpha = 0.5, color = 'red')
    axs[1,0].set_title('Z q(k)')
    axs[1,1].plot(x_xrot, y_xrot, lw=2, alpha = 0.5, color = 'violet')
    axs[1,1].set_title('Xrot q(k)')
    axs[1,2].plot(x_yrot, y_yrot, lw=2, alpha = 0.5, color = 'magenta')
    axs[1,2].set_title('Yrot q(k)')
    axs[2,0].plot(x_zrot, y_zrot, lw=2, alpha = 0.5, color = 'purple')
    axs[2,0].set_title('Zrot q(k)')
    axs[2,1].plot(x_sex, y_sex, lw=2, alpha = 0.5, color = 'cyan')
    axs[2,1].set_title('Sex q(k)')
    #pass it to TB writer
    writer.add_figure("q(k)_{}".format(log_type), fig)

def log_beta(writer, xq, beta_mean, beta_cov, covariate_name, log_type):
    """
    Logs beta dist plots to TB.
    This is done from within fwd method.
    """
    cov_dict = {}
    xq = xq.cpu().numpy()
    beta_mean = beta_mean.detach().cpu().numpy()
    two_sigma = 2*np.sqrt(np.diag(beta_cov.detach().cpu().numpy()))
    cov_dict['xq'] = xq
    cov_dict['mean'] = beta_mean
    cov_dict['two_sig'] = two_sigma
    cov_data = pd.DataFrame.from_dict(cov_dict)
    sorted_cov_data = cov_data.sort_values(by=["xq"])
    fig = plt.figure()
    plt.plot(sorted_cov_data['xq'], sorted_cov_data['mean'], \
    c='darkblue', alpha=0.5, label='Beta posterior mean')
    kwargs = {'color':'lightblue', 'alpha':0.3, 'label':'2 sigma'}
    plt.fill_between(sorted_cov_data['xq'], (sorted_cov_data['mean'] - sorted_cov_data['two_sig']), \
    (sorted_cov_data['mean'] + sorted_cov_data['two_sig']), **kwargs)
    plt.legend(loc='best')
    plt.title('Beta_{}'.format(covariate_name))
    plt.xlabel('Covariate')
    plt.ylabel('Beta Ouput')
    writer.add_figure("Beta/{}_{}".format(covariate_name, log_type), fig)

def log_map(writer, img_shape, map, slice, map_name, batch_size, log_type):
    """
    Logs a particular brain map reconstruction to TB.
    Args
    ----
    Map: (np array) map reconstructions for a given minibatch.
    slice: (int) specific slice we wish to log.
    map_name: (string) Name of map (e.g., base, task)
    batch_size: (int) Size of minibatch.
    For now am logging slices only in saggital view.
    """
    map = map.reshape((batch_size, img_shape[0], img_shape[1], img_shape[2]))
    for i in range(batch_size):
        slc = map[i, slice, :, :]
        slc = ndimage.rotate(slc, 90)
        fig_name = '{}_{}_{}/{}'.format(map_name, log_type, slice, i)
        writer.add_image(fig_name, slc, dataformats='HW')

#####################
#Helpers for OLS map creation
####################

def read_design_mat(mat_file_path):
    """
    Reads Design Matrix Files generated by FSL's feat module.
    These are used to find Least Squares Solution to beta maps
    which is then used to regularize model.

    ToDo's: get rid of a, b, c steps here...
    This is poor coding :(
    """
    with open(mat_file_path) as f:
        content = f.readlines()
    design_mat = []
    for i in range(5, len(content)):
        a = content[i].rstrip()
        b = re.split(r'\t+', a)
        c = [float(i) for i in b]
        design_mat.append(c)
    design_mat = np.array(design_mat)
    return design_mat

def scale_beta_maps(beta_maps):
    """
    Performs min-max scaling for least squares maps used in regularization.
    This helps regularizer portion of loss (and overall model) to behave better.
    """
    for i in range(beta_maps.shape[0]):
        map_max = np.amax(beta_maps[i, :].flatten())
        beta_maps[i, :] = beta_maps[i, :]/map_max
    return beta_maps

def get_all_runs_data(feat_dirs, subj_idx, data_dims, num_runs=4):
    """
    Constructs array containing filtered data for all runs for a given subj.
    ---
    Args:
    feat_dirs: dict containing all .feat dirs for each subj. Keys are subj ids.
    subj_idx: index for subj we wish to get concatenated filtered data for.
    num_runs: number of runs each subj is supposed to have. For EMERALD this is 4.
    """
    subjs = list(feat_dirs.keys())
    all_runs_data = []
    for i in range(num_runs):
        run_filt_data_path = os.path.join(feat_dirs[subjs[subj_idx]][i], 'filtered_func_data.nii.gz')
        assert os.path.exists(run_filt_data_path), 'Failed to find filtered data for run {}'.format(i)
        run_filtered_data = np.array(nib.load(run_filt_data_path).dataobj).reshape(data_dims[0]*data_dims[1]*data_dims[2], -1)
        all_runs_data.append(run_filtered_data)
    filtered_data = np.concatenate(all_runs_data, axis=1)
    return filtered_data

def get_all_runs_dms(feat_dirs, subj_idx, data_dims, num_runs=4):
    """
    Reads and concatenates DMs for all runs for a given subj.
    ---
    Args:
    feat_dirs: dict containing all .feat dirs for each subj. Keys are subj ids.
    subj_idx: index for subj we wish to get concatenated filtered data for.
    num_runs: number of runs each subj is supposed to have. For EMERALD this is 4.
    """
    subjs = list(feat_dirs.keys())
    all_dms = []
    for i in range(num_runs):
        run_mat_path = os.path.join(feat_dirs[subjs[subj_idx]][i], 'design.mat')
        assert os.path.exists(run_mat_path), 'Failed to find design matrix for run {}'.format(i)
        run_mat = read_design_mat(run_mat_path)
        #take first 3 cols of DM --> these are the main EMERALD cons we care about
        task_cols = run_mat[:, 0:3].reshape((-1, 3))
        #am NOT taking motion cols b/c they did original analysis with outlying vols per run/subj
        #this changes for each run/subj (in number of motion covs/maps)
        all_dms.append(task_cols)
    gamma = np.concatenate(all_dms, axis=0)
    return gamma

def get_OLS_sln(gamma, filtered_data):
    """
    Computes OLS sln to GLM using concatenated DMs and filtered data.
    This should be done for each subj or as a streaming/running estimate.
    ---
    Args:
    gamma: matrix with task covariates and motion confounders.
    filtered_data: matrix with concatenated filtered data for all runs.
    """
    pseudo_inv = np.linalg.inv(np.matmul(gamma.T, gamma))
    pseudo_inv = np.matmul(pseudo_inv, gamma.T)
    beta_maps = np.matmul(pseudo_inv, filtered_data.T)
    return beta_maps
