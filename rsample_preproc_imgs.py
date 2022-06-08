"""
Short script to resample EMERALD data preprocessed functional imgs
into shape needed for VAE-GAM network.

In future, I might simply change network sizes if this makes a difference - tbs.
"""

import os
import subprocess

###########################################################
# def some helpers
#these are the same as in JG's original preprocessing scripts
##########################################################

def __add_prefix(input_file, prefix):
    #This function appends a string to the existing prefix of an image file.
    #It assumes the image file is either .nii or .nii.gz.
    input_beginning, input_end = input_file.split('.nii')
    output_file = input_beginning+str(prefix)+'.nii'+input_end
    return output_file

def __resampling(input_image, output_image, overwrite = 0, skip = 0):
    #This function will apply the downsampling step at the end of post-fmriprep after masking.
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
            print('No output_image passed, will append "_no_output_image" to the input_image name!')
            output_file = __add_prefix(input_file, '_no_output_image')
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
        reference = "/mnt/keoki/experiments2/Rachael/data/emo_class/fmri_data/subject201films_20110509_12861/subject201films_all_runs.nii"
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

###################################################
#Now do resampling for all subjs and runs in cohort
###################################################

good_subjs = []
failed_subjs = []

#specify subs and runs to apply rsampling to
#could also specify task and ses (but we don't need it here)
subs_to_run = ['0001', '0036', '0071', '0162', '0174', '0179', '0038', '0088', '0126', '0155', '0184', '0187']
runs_to_run = ['1', '2', '3', '4']

#below is already preprocessed, with first 4TRs cutout, and with smoothing, temporal filtering and brain masking applied
gen_input_path = '/mnt/keoki/experiments2/VAE_GAM/Data/EMERALD_cohort/fmriprep/sub-EM{s}/func/sub-EM{s}_emoreg_run{r}_preproc_short_tempfilt_smooth_brain.nii.gz'
gen_output_path = '/mnt/keoki/experiments2/VAE_GAM/Data/EMERALD_cohort/fmriprep/sub-EM{s}/func/sub-EM{s}_emoreg_run{r}_preproc_short_tempfilt_smooth_brain_rsampled.nii.gz'

for sub in subs_to_run:
    for run in runs_to_run:
        sub_run_input_img = gen_input_path.format(s=sub, r=run)
        sub_run_output_img = gen_output_path.format(s=sub, r=run)
        try:
            rsampled_img = __resampling(sub_run_input_img, output_image = sub_run_output_img, overwrite=0)
            if rsampled_img is None:
                raise RuntimeError('Resampling')
            good_subjs.append([sub, run])
        except Exception as ex:
            failed_subjs.append([sub, run, ex])

print('---------------------------')
print('Good subjs: {}'.format(good_subjs))
print('Bad subjs: {}'.format(failed_subjs))
