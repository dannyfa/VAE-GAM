"""
Post-processing module --> to be called from wrapper script after model has been
trained and reconstructions, GP plots and latent space plots have been created.

Purpose: makes GP means centered around zero by adding maps for
constant GP regressors to base map, leaving only GPs with non-cte predictions
as separate covariate effects/maps.

Args:
- Analysis directory: this is the same as --save_dir in the wrapper script
- Model (a VAE-GP model object)
- Cutoff: cutoff for yq variance value. Covariates with yq vars <= cutoff
will be merged to base maps.

Output:
Writes new base maps to --save_dir under a new sub-directory
named (/reconstructions/#epochs_avg_model_recons/post_processed).
New base maps are created for both gran_avg maps (i.e., across subjects) and
for subj_avg_maps.
Also prints/writes:
1) which covariates were merged to base
2) which cutoff was used
3) which GP summary file (named as, #epochs_GP_yq_variances.csv) was used
"""

import numpy as np
import nibabel as nib
import os, sys
import pandas as pd
import datetime


#define some useful helper functions first

def _check_needed_maps (dir, needed_files_list):
    dir_contents = os.listdir(dir)
    for f in needed_files_list:
        task_f = f + '_avg.nii'
        null_f = 'null_' + f + '_avg.nii'
        if not task_f in dir_contents:
            print("Missing an expected avg map: {}".format(task_f))
            sys.exit()
        if not null_f in dir_contents:
            print("Missing an expected avg map: {}".format(null_f))
            sys.exit()
    print(40*'=')
    print("All needed files found successfully under {}".format(dir))

def _save_map(map, reference, save_dir, ext):
    """
    Helper function for mk_avg_maps
    Takes an array corresponding to a regressor map, a reference nifti file and
    a saving directory and outputs a saved nifti file corresponding to regressor map.
    Only needed b/c I am using nibabel and nifti format.
    This might be taken out entirely in future if we decide to use other libraries or
    file formats like  hdr.
    """
    ref = nib.load(reference)
    nii = nib.Nifti1Image(map, ref.affine, ref.header)
    path = os.path.join(save_dir, '{}_avg.nii'.format(ext))
    nib.save(nii, path)

def _merge_maps(dir, covariates_to_merge):
    base_path = os.path.join(dir, 'base_avg.nii')
    nullbase_path = os.path.join(dir, 'null_base_avg.nii')
    base_map = np.array(nib.load(base_path).dataobj)
    nullbase_map = np.array(nib.load(nullbase_path).dataobj)
    for cov in covariates_to_merge:
        cov_map_path = os.path.join(dir, (cov + '_avg.nii'))
        nullcov_map_path = os.path.join(dir, ('null_' + cov + '_avg.nii'))
        cov_map = np.array(nib.load(cov_map_path).dataobj)
        nullcov_map = np.array(nib.load(nullcov_map_path).dataobj)
        base_map += cov_map
        nullbase_map += nullcov_map
    #save new 'merged' base
    out_path = os.path.join(dir, 'post_processed')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    _save_map(base_map, base_path, out_path, 'postproc_base')
    _save_map(nullbase_map, base_path, out_path, 'postproc_nullbase')

#now run postproc per se
def run_postproc (model, analysis_dir, cutoff):
    #get num epochs from loaded/trained model
    epochs = str(model.epoch).zfill(3)
    #get gP_summary_file path & mk sure it exists
    gp_summary_path = os.path.join(analysis_dir, \
    (epochs + '_GP_plots'), (epochs + '_GP_yq_variances.csv'))
    if not os.path.exists(gp_summary_path):
        print("Could not find GP summary file for specified model/#epochs!")
        print("Make sure to run model.plot_GPs prior to running zeroGPmeans routine")
        sys.exit()
    print(40*'=')
    print("Using gp_summry file on: {}".format(gp_summary_path))
    print("Cutoff set to: {}".format(cutoff))
    #get dir with avg model recons & mk sure it exists
    avg_recons_dir = os.path.join(analysis_dir, 'reconstructions', \
    (epochs + '_avg_model_recons'))
    if not os.path.exists(avg_recons_dir):
        print("Could not find directory containing avg reconstruction maps!")
        print("Make sure to run build_model_recons prior to runnign zeroGPmeans routine")
        sys.exit()
    #check that all necessary files exist
    #this has to be done both for gd_avg maps
    #and subj level maps
    avg_dir_contents = os.listdir(avg_recons_dir)
    subjs = [i for i in avg_dir_contents if 'sub-A000' in i]
    needed_files = ['base', 'task', 'x_mot', 'y_mot', 'z_mot', \
    'pitch_mot', 'roll_mot', 'yaw_mot']
    subj_recons_dirs = []
    #check gd_avgs first
    _check_needed_maps(avg_recons_dir, needed_files)
    #check subj level dirs too
    for subj in subjs:
        subj_dir = os.path.join(avg_recons_dir, subj)
        subj_recons_dirs.append(subj_dir)
        _check_needed_maps(subj_dir, needed_files)
    #now read summary file & get covars to be merged
    df = pd.read_csv(gp_summary_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')] #drop any unnamed cols
    covariates_to_merge = []
    for col in df.columns:
        if df[col].iloc[0] <= cutoff:
            covariates_to_merge.append(col)
    print("Merging base with following covariates : {}".format(covariates_to_merge))
    #now perform merge and save resulting Maps
    _merge_maps(avg_recons_dir, covariates_to_merge)
    for subj_dir in subj_recons_dirs:
        _merge_maps(subj_dir, covariates_to_merge)

    #write summary file to facilitate bookeeping of analysis
    ts = datetime.datetime.now().date()
    out_path = os.path.join(avg_recons_dir, 'post_processed', \
    ('postproc_summary' + '_' + ts.strftime('%m_%d_%Y') + '.txt'))
    f = open(out_path, "w")
    f.write("Analysis dir used: {} \n".format(analysis_dir))
    f.write('Cutoff used: {:.2e} \n'.format(cutoff))
    f.write('GP_summary_file used: {}\n'.format(gp_summary_path))
    f.write('Covariates merged: {}\n'.format(covariates_to_merge))
    f.close()
