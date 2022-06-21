"""
Short script to resample EMERALD data preprocessed functional imgs
into shape needed for VAE-GAM network.

In future, I might simply change network sizes if this makes a difference - tbs.

Am leaving this as a separate script b/c this makes it easier to run it separately in the LaBar server.
"""

import os
import subprocess

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
    #essentially this is a wrapper for afni's 3dresample function.
    print('-------Starting: Resampling-------')
    try:
        #Check the input file for a path
        input_path, input_file = os.path.split(input_image)
        if input_path is '':
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


#start logs for good and failed runs
good_subjs = []
failed_subjs = []

#specify subs and runs to apply rsampling to
#these should be args in future
subs_to_run = ['0187']
runs_to_run = ['1', '2', '3', '4']

#def general path name/root
#and ref nifti for resampling
#these will also be args in the future
gen_input_path = '/mnt/keoki/experiments2/VAE_GAM/Data/EMERALD_cohort/fmriprep/TEST_set/sub-EM{s}/func/sub-EM{s}_emoreg_run{r}_preproc_short_tempfilt_smooth_brain.nii.gz'
ref = "/mnt/keoki/experiments2/Rachael/data/emo_class/fmri_data/subject201films_20110509_12861/subject201films_all_runs.nii"

for sub in subs_to_run:
    for run in runs_to_run:
        sub_run_input_img = gen_input_path.format(s=sub, r=run)
        try:
            rsampled_img = resampling(sub_run_input_img, ref, output_image = None, overwrite=1)
            if rsampled_img is None:
                raise RuntimeError('Resampling')
            good_subjs.append([sub, run])
        except Exception as ex:
            failed_subjs.append([sub, run, ex])

print('---------------------------')
print('Good subjs: {}'.format(good_subjs))
print('Bad subjs: {}'.format(failed_subjs))
