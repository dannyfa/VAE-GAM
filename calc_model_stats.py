"""
This script adopts per voxel, frequentist approach
We will likely chnage it entirely to something else!!

Short script to calculate TP, TN, FP, FN rates AND
other associated stats for our model, namely:
error = (FP + FN)/(TP + TN + FP + FN)
accuracy = (TN + TP)/(TP + TN + FP + FN)
sensitivity = TP/(TP + FN)
specificity = TN / (TN + FP)
precision = TP/(TP + FP)
false positive rate = FP / (FP + TN)
false negative rate = FN / (FN + TP)
true positive rate = TP/(TP + FN)

Takes in paths to 2 maps:
1) Ground-truth map (either a know signal -- i.e., if running for controls) or a map produced by std software (e.g., FSL)
2) Experiemental map (across-subjs avg map from our model)

As of rn does NOT support single subj stats or stats for each volume (i.e., for different pts in time)

Thresholding criteria used to generate binary masks of gt and experiment is the SAME and follows the convention (mean + x*stds),
where x is some user input (float) and std is std deviation of each map.

For convinience, binary masks are saved to output dir -- this allows user to check if threshold utilized is appropriate.
"""

import numpy as np
import nibabel as nib
import argparse
import os, sys
import datetime

#get user args
parser = argparse.ArgumentParser(description='user args for calculating model stats')

parser.add_argument('--gt', type=str, metavar='N', default='', \
help='Path to ground-truth map used in this analysis.')
parser.add_argument('--exp', type=str, metavar='N', default='', \
help='Path to experimental grand-avg map used in this analysis.')
parser.add_argument('--cutoff', type=float, metavar='N', default=6.0, \
help='Thresholding cutoff - corresponds to number of stds above mean needed for inclusion in bin mask.')
parser.add_argument('--out_dir', type=str, metavar='N', default='', \
help='Output dir where binary masks and stats summary is saved to.')
parser.add_argument('--verbose', type=bool, metavar='N', default= False, \
help='If set to True, will print out additional information like map descriptive stats.')

args = parser.parse_args()

#mk sure user inputs are ok!

if args.gt == '':
    print("Cannot proceed without ground-truth map!")
    sys.exit()
if not os.path.exists(args.gt):
    print("Ground-truth path does NOT exist!")
    sys.exit()

if args.exp == '':
    print("Cannot proceed without experimental map!")
    sys.exit()
if not os.path.exists(args.exp):
    print("Experimental map path does NOT exist!")
    sys.exit()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


#define helper function to save files
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
    path = os.path.join(save_dir, '{}_bin_mask.nii'.format(ext))
    nib.save(nii, path)

#get gt and experimental maps
#get shape of these arrays
gt_map = np.array(nib.load(args.gt).dataobj)
exp_map = np.array(nib.load(args.exp).dataobj)
x, y, z = gt_map.shape[0], gt_map.shape[1], gt_map.shape[2]
#get mean and atd for these...
gt_mean, gt_std = np.mean(gt_map.flatten()), np.std(gt_map.flatten())
exp_mean, exp_std = np.mean(exp_map.flatten()), np.std(exp_map.flatten())
if args.verbose == True:
    print(40*'=')
    print('Ground-truth map descriptive stats are:\n')
    print('mean: {:.2f}, std: {:.2f}'.format(gt_mean, gt_std))
    print(40*'=')
    print('Experimental map descriptive stats are:\n')
    print('mean: {:.3f}, std: {:.3f}'.format(exp_mean, exp_std))
#now threshold maps & turn them into binary masks
masked_gt = np.where(gt_map.flatten() > (gt_mean + 6*gt_std), 1, 0).reshape(x, y, z)
masked_exp = np.where(exp_map.flatten() > (exp_mean + args.cutoff*exp_std), 1, 0).reshape(x, y, z)
#save these binary masks
_save_map(masked_gt, args.gt, args.out_dir, 'gt')
_save_map(masked_exp, args.exp, args.out_dir, 'exp')
#now compute statistics
#am  using diff between 2*masked_gt and masked_exp to get tn, tp, fp and fn counts
#note that:
#if diff ==1, TP
#if diff == 0, TN
#if diff == -1, FP
#if diff == 2, FN
diff_map = 2 * masked_gt.flatten() - masked_exp.flatten()
tp = np.sum(diff_map ==1)
tn = np.sum(diff_map ==0)
fp = np.sum(diff_map ==-1)
fn = np.sum(diff_map ==2)
if args.verbose == True:
    print(40*'=')
    print('Here are my counts:')
    print('tn: {}, tp: {}, fp: {}, fn: {}'.format(tn, tp, fp, fn))
error_rate = (fp + fn)/(tp + tn + fp + fn)
accuracy = (tp + tn)/(tp + tn + fn + fp)
sensitivity = tp/(tp + fn)
specificity = tn/(tn + fp)
precision = tp/(tp + fp)
fpr = fp/(fp + tn)
tpr = tp/(tp + fn)
fnr = fn/(fn + tp)
#now save these computed stats to a file
ts = datetime.datetime.now().date()
out_path = os.path.join(args.out_dir, ('summary' + '_' + ts.strftime('%m_%d_%Y') + '.txt'))
f = open(out_path, "w")
f.write('Summary stats for: \n')
f.write('Ground-truth: {}\n'.format(args.gt))
f.write('VAE avg: {}\n'.format(args.exp))
f.write (40*'#')
f.write ('\n')
f.write('Count summaries:\n')
f.write('tp: {}, tn: {}, fp: {}, fn: {}\n'.format(tp, tn, fp, fn))
f.write (40*'#')
f.write ('\n')
f.write('Stats summaries: \n')
f.write('error_rate: {:.3f}, accuracy: {:.3f}\n'.format(error_rate*100, accuracy*100))
f.write('sensitivity: {:.3f}, specificity: {:.3f}\n'.format(sensitivity*100, specificity*100))
f.write('precision: {:.3f}, false positive rate: {:.3f}, false negative rate: {:.3f}, true positive rate: {:.3f}\n'.format(precision*100, fpr*100, fnr*100, tpr*100))
f.close()
