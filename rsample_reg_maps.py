"""
Short script to resample reg maps constructed for EMERALD data analysis
into shape needed for VAE-GAM network.

Again am making this script separate and really NOT optimal for sake of being able to use afni
only in the LaBar server (instead of installing it in lab machines).

Might be worth just installing afni and being happy about it later on...

"""
import os
import subprocess
import pandas as pd
import nibabel as nib
import numpy as np

###############################
#Helpers for 3D resampling step
################################

def __add_prefix(input_file, prefix):
    #This function appends a string to the existing prefix of an image file.
    #It assumes the image file is either .nii or .nii.gz.
    input_beginning, input_end = input_file.split('.nii')
    output_file = input_beginning+str(prefix)+'.nii'+input_end
    return output_file

def resampling(input_image, reference, output_image=None, overwrite = 0, skip = 0):
    #This function will apply a resampling opp to preprocessed fMRI data
    #essentially this is a wrapper for afni's rsampling function.
    print('-------Starting: Resampling-------')
    try:
        #Check the input file for a path
        input_path, input_file = os.path.split(input_image)
        if input_path == '':
            print('input_image must contain a full path to the image file!')
            raise RuntimeError()

        #Check that the input file is either a .nii or .nii.gz file
        if len(input_file.split('.nii')) == 1:
            print('input_image file type not recognized. Should be .nii or .nii.gz!')
            raise RuntimeError()

        #Put together the output file
        if output_image is None:
            print('No output_image passed, will append "_rsample" to the input_image name!')
            output_file = __add_prefix(input_file, '_rsampled')
            output_image = os.path.join(input_path, output_file)

        #Check to see if passed output image is already there
        if os.path.exists(output_image):
            print('output_image already exists!')
            if skip:
                print('Skip set, returning...')
                return output_image
            if overwrite:
                print('Overwrite set to 1, deleting...')
                os.remove(output_image)
            else:
                print('Overwrite not set, exitting...')
                return None

        resampling_call = [
            '3dresample',
            '-master',
            reference,
            '-prefix',
            output_image,
            '-inset',
            input_image
        ]
        print('Resampling Image...')
        os.system(' '.join(resampling_call))

        if not os.path.exists(output_image):
            print('output_image should be there, but is not: {}'.format(output_image))
            return None
    except:
        print('ERROR in resampling image!')
        return None

    print('Image resampling successful.')
    print('------Done: resampling------')
    return output_image


#read input file w/ scaled maps and write these to Nifti's
#then rsample -- will output nifti's again
#read these back and put them into desired csv format
csv_input = '/mnt/keoki/experiments2/VAE_GAM/Data/EMERALD_cohort/EMERALD_test_06162022/scld_GLM_beta_maps.csv'
rsampling_ref = "/mnt/keoki/experiments2/Rachael/data/emo_class/fmri_data/subject201films_20110509_12861/subject201films_all_runs.nii"
rsampling_ref_dims = [41, 49, 35]
orig_ref_path = '/mnt/keoki/experiments2/VAE_GAM/Data/EMERALD_cohort/Analysis/TRAIN_set/sub-EM0001/First_level_run1.feat/filtered_func_data.nii.gz'
orig_ref = nib.load(orig_ref_path)
orig_img_shape = [91, 109, 91] #3D shape for orig data
output_root = '/mnt/keoki/experiments2/VAE_GAM/Data/EMERALD_cohort/EMERALD_test_06162022/'

reg_maps = pd.read_csv(csv_input, index_col=False).to_numpy()

rsampled_maps = []

for i in range(1, reg_maps.shape[1]):
    #save original file to nifti -- so that afni can read it
    reg_map = np.squeeze(reg_maps[:, i].reshape(-1, orig_img_shape[0], orig_img_shape[1], orig_img_shape[2]))
    nifti_img = nib.Nifti1Image(reg_map, orig_ref.affine, orig_ref.header)
    out_path = os.path.join(output_root, 'reg_map_{}.nii.gz'.format(i))
    nib.save(nifti_img, out_path)

    #do rsampling step
    rsampled_img = resampling(out_path, rsampling_ref, output_image=None, overwrite=1)

    #read file again
    #and concatenate it appropriately
    rsampled_map = np.array(nib.load((out_path[:-7] + '_rsampled.nii.gz')).dataobj)
    rsampled_maps.append(rsampled_map.reshape(-1, rsampling_ref_dims[0]*rsampling_ref_dims[1]*rsampling_ref_dims[2]))
rsampled_maps = np.concatenate(rsampled_maps, axis=0)

#write final rsampled csv
#this is what we will feed into model ultimately
rsampled_beta_maps_df = pd.DataFrame(rsampled_maps.T, columns = ['flow', 'reappraisal', 'distancing'])
rsampled_beta_maps_df.to_csv(os.path.join(output_root, 'rsampled_scld_GLM_beta_maps.csv'))
