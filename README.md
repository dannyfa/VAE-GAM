# VAE-GAM

PyTorch implementation of VAE-GAM model for task-based fMRI analysis described in: https://proceedings.mlr.press/v149/albuquerque21a <br>

This model provides a more flexible means to analyze task-based fMRI data stemming from designed experiments in cognitive, systems
and clinical neuroscience. <br>

To get started using this code, clone this repository and make sure you have all the dependencies listed under <em> dependencies.txt </em> installed. <br>

Scripts should allow users to: <br>
1)Create required files for model training.<br>
2)Create synthetic data for synthetic signal simulations demonstrated in paper. <br>
3)Train VAE-GAM model on either synthetic or real data. <br>
4)Generate latent space plots, plots for Gaussian Process regressors and single volume reconstructions,
as well as subject-level and group-level average maps for covariate, base and for
full-reconstruction maps. <br>

Pre-processing scripts assume user has <strong> ALREADY </strong> performed some basic pre-processing on his/hers fMRI dataset using standard neuroimaging software (e.g., FSL).
For the experiments showcased in paper, we utilized fmriprep for the preprocessing steps. Additional details on specific routines/parameters used for fMRI preprocessing can be provided upon
request, along with the preprocessed nifti files we used to run the checker experiments/simulations.

To generate the csv_file to be used by DataClass and Loaders: <br>
<br>
<strong> 1)Run pre_proc_vaefmri.py </strong> <br>
<br>
  <em> python pre_proc_vaefmri.py --data_dir {dir w/ your preprocessed fMRI data} --save_dir {dir where you want your csv_file saved to} <br>
  --nii_file_pattern {your_nii_filename_pattern} --mot_file_pattern {your_mot_filename_pattern} </em> <br>

  The last 2 flags refer to filename patterns for your pre-processed fMRI data (assumes nifti format!) and for the motion files generated during preprocessing. <br>

  User can also choose to use flag <em> '--control' </em> to generate csv_file for a control experiment simulation. In this case, user should pass intensity of synthetic control signal present in data using the <em> --control_int </em> flag. Finally, this program also adds either a 'TRAIN' or 'TEST' tag to the generated preprocessed filename, indicating if csv corresponds to train or test set. Default is 'TRAIN' - to generate csv for test set run above command with <em> --set_tag </em> as 'TEST'.

 <strong> 2)Run get_beta_map_regularizer.py. </strong> <br>
 <br>
  This script will generate a rough (least-squares estimate) map using the preprocessed fMRI data and their corresponding design matrices (produced using std GLM software like FSL). This rough map is used as a regularizer, so as to encourage our model to produce maps that are not too far off from main effects expected using GLM approach. This regularizer was not needed when running control experiments, only for actual biological signals like V1 signal/experiments showcased in paper. <br>
  <br>
  <em> python get_beta_map_regularizer.py --root_dir {dir w/ pre-processed fMRI data} --output_dir {dir where you want the lsqrs map to.} --data_dims {x, y, z, time - these are the dimensions for 4D fMRI data.}<br></em>

 <strong>  3)Train model </strong> <br>
 <br>
 <em> python multsubj_reg_run_GP.py --train_csv {csv file for train set} --test_csv {csv file for test set} --save_dir {dir where model checkpoints, latent space plots, GP plots and
 and reconstructions/maps will be saved to} --glm_maps {path to lstsqrs map generated in step 2 above.}</em> <br>

 User can train model using a pre-existing checkpoint file. IF this behavior is desired simply add the flag <em> --from_ckpt </em> to the command above and give
 location of checkpoint file to be used <em> --ckpt_path {my_ckpt} </em>. <br>
 <br>
 User can also choose to simply generate latent space plots, GP plots and map reconstructions from a pre-existing checkpoint file (and without training model
 further). If this is desired add <em> --recons_only </em> flag to command above, along with <em> --from_ckpt </em> flag and with location for
 ckpt file to be used when creating model outputs <em> --ckpt_path {my_ckpt} </em>. <br>
 <br>
 You may wish to change other model parameters such as weights for the gp KL regularization, weights for the lstsqrs map regularizer or number of inducing points for the GP regressors. You may do so
 by changing values of <em> --gp_kl_scale </em> , <em> --glm_reg_scale </em> and <em> --num_inducing_pts </em> respectively. However, we do not advise doing so unless you fully understand rest of code.
 <br>

 Finally this script also takes a <em> --neural_covariates </em> flag, which indicates wether coveriates passed are real/bilogical or synthetic. Default is 'True', meaning code will treat covariates passed as being real biologically-relevant signals, which will be convolved with HRF. Note that last 6 covariates passed in csv file are assumed to be motion-related nuisance covariates. These are NEVER convolved with the HRF (regardless of choice for this flag).
<br>
<br>

<strong> 4)Adding synthetic signals to existing data. </strong> <br>
  <br>
To construct data sets with the synthetic signals shown in paper, run the following command: <br>
<br>
<em> python add_control_signal.py --root_dir {dir with preprocessed fMRI data we wish to add synthetic signal to} --intensity {intensity of added singal} --shape {'Large3'} --nii_file_pattern {filename pattern for nifti files under root_dir to be used.}</em><br>
<br>
Of note, this script <strong> WILL NOT </strong> overwrite the data under <em> --root_dir </em>.
Instead, it will write data with synthetic signal to <em> --root_dir </em> with same name as original + suffix 'ALTERED_Large3_{intensity}_simple_ts_{date_stamp}.nii.gz' <br>

For any questions regarding this repository, the paper, replicating our simulations or extending this work please contact Daniela de Albuquerque -- dfd4@duke.edu.
<br>
