# VAE-GAM
Repo for fMRI VAE-GAM model scripts 

PyTorch implementation of VAE-GAM model for task-based fMRI analysis described in: https://www.biorxiv.org/content/10.1101/2021.04.04.438365v2.abstract <br>

This model provides a more flexible and potentially powerful means to analyze task-based fMRI data stemming from designed experiments in cognitive, systems 
and clinical neuroscience. <br>

To get started using this code, clone this repository and make sure you have all the dependencies listed under <em> dependencies.txt </em> installed. <br>

Scripts should allow users to: <br>
1)Create required files for model training.<br>
2)Create data for synthetic signal simulations demonstrated in paper. <br>
3)Train VAE-GAM model on either synthetic or real data. <br>
4)Generate latent space plots, plots for Gaussian Processes regressors and single volume reconstructions, 
as well as subject-level and group-level average maps for each covariate, for base and for 
full volume. <br>

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
  
  User can also choose to use flag <em> '--control' </em> to generate csv_file for a control experiment simulation. 
  
 <strong> 2)Run calc_SPM_GLM_avgs.py. </strong> <br>
 <br>
  This script will generate an average map out of 'n' single-subject GLM maps created with your preferred standard fMRI analyses software. The resulting map 
  can be used to initialize the model. This is not strictly necessary for control simulations, BUT yielded best results (reported in manuscript) 
  for checker simulations. <br>
  <br>
  <em> python calc_SPM_GLM_avgs.py --data_dir {w/ GLM analyzed fMRI data} --save_dir {dir where you want the avg map saved to} <br>
  --ref_nii {reference nifti file to be used. Can be any of the files to be averaged -- as long as these have been warped to same space!}. </em> <br>
  
  Of note, will use all files under --data_dir to create avg. In our experiments, we ONLY did an average of two random GLM maps in our sample. 
  You should <strong> NOT </strong> need to run a GLM for your entire sample to obtain good results!
  
 <strong>  3)Train model </strong> <br>
 <br>
 <em> python multsubj_reg_run_GP.py --csv_file {your csv_file from step #1} --save_dir {dir where model checkpoints, latent space plots, GP plots and 
 and reconstructions/maps will be saved to} --task_init {path to avg map constructed in step #2}</em> <br>
 
 User can train model using a pre-existing checkpoint file. IF this behavior is desired simply add the flag <em> --from_ckpt </em> to the command above and give 
 location of checkpoint file to be used <em> --ckpt_path {my_ckpt} </em>. <br>
 <br>
 User can also choose to simply generate latent space plots, GP plots and map reconstructions from a pre-existing checkpoint file (and without training model
 further). If this is desired add <em> --recons_only </em> flag to command above, along with <em> --from_ckpt </em> flag and with location for 
 ckpt file to be used when creating model outputs <em> --ckpt_path {my_ckpt} </em>. <br>
 <br>
 You may wish to change other model parameters such as weight for GP_loss component or weight for l1 regularization imposed on all covariate maps. You may do so 
 by changing values of <em> --mll_scale </em> and <em> --l1_scale </em>. However, we do not advise doing so unless you fully understand rest of code. 
 <br>
<br>
<strong> 4)Adding synthetic signals to existing data. </strong> <br>
  <br>
To construct data sets with the synthetic signals shown in paper, run the following command: <br>
<br>
<em> python add_control_signal.py --root_dir {dir with preprocessed fMRI data we wish to add synthetic signal to} --intensity {intensity of added singal} </em><br>
<br>
Of note, this script <strong> WILL NOT </strong> overwrite the data under <em> --root_dir </em>. 
Instead, it will write data with synthetic signal to <em> --root_dir </em> with same name as original + suffix 'ALTERED_Large3_{intensity}_simple_ts_{date_stamp}' added
to it. <br>

For any questions regarding this repository, the paper, replicating our simulations or extending this work please contact Daniela de Albuquerque -- dfd4@duke.edu. 
<br>
