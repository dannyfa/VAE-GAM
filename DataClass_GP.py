"""
Script defining fMRIDataset Class and loaders to be used along with VAE-GAM model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import nibabel as nib

class FMRIDataset(Dataset):
    """Defines FMRIDataset Class to be used with VAE-GAM model. """
    def __init__(self, csv_file, transform= None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        """Return the number of samples in dset"""
        return len(self.df)
    def __getitem__(self, idx):
        """Returns a single sample from dset.
           Each sample is a dict w/ following keys:
            subj: unique ID (string) for each subj. Repeats across vols for same subj.
            subj_idx: index (int) for each subj in dset. Repeats across vols for the same subj.
            volume: np array (3D) containing a (globally scaled) volume from a given subj.
            vol_num: int representing volume # (in acquisition order).
            task: binary task variable. This will be convolved with HRF during training for neural/biological covariates.
            trans_x, trans_y, trans_z: z-scored head translation in x, y, z axes respectively.
            rot_x, rot_y, rot_z: head rotation over 3 axes.
        """
        #get subjid and its index
        unique_subjs = self.df.subjid.unique().tolist()
        subj = self.df.iloc[idx,1]
        subj_idx = unique_subjs.index(subj)
        #get all other covariates
        vol_num = self.df.iloc[idx,2]
        nii = self.df.iloc[idx,3]
        task = self.df.iloc[idx,4]
        #motion params
        trans_x = self.df.iloc[idx,5]
        trans_y = self.df.iloc[idx,6]
        trans_z = self.df.iloc[idx,7]
        rot_x = self.df.iloc[idx,8]
        rot_y = self.df.iloc[idx,9]
        rot_z = self.df.iloc[idx,10]
        #gender/sex
        sex = self.df.iloc[idx, 11]

        fmri = np.array(nib.load(nii).dataobj)
        max = 3284.5 #across all vols. Used to scale data (globally).
        volume = fmri[:,:,:,vol_num]
        flat_vol = volume.flatten()
        scld_vol = np.true_divide(flat_vol, max).reshape(41,49,35)
        #now create sample.
        sample = {'subj_idx': subj_idx, 'subj': subj, 'volume': scld_vol, 'vol_num':vol_num,
                  'task':task, 'trans_x':trans_x, 'trans_y':trans_y, 'trans_z':trans_z,
                  'rot_x':rot_x, 'rot_y':rot_y, 'rot_z':rot_z, 'sex':sex}
        if self.transform:
            sample = self.transform(sample)
        return(sample)

class ToTensor(object):
    "Converts sample arrays to tensors which can be directly fed to model."
    def __call__(self, sample):
        subjid, volume, vol_num = sample['subj_idx'], sample['volume'], sample['vol_num']
        #Concat task w/ mot params by row
        covars = np.array([sample['task'], sample['trans_x'], sample['trans_y'], \
        sample['trans_z'], sample['rot_x'], sample['rot_y'], sample['rot_z'], sample['sex']], dtype=np.float64)
        return{'covariates':torch.from_numpy(covars).float(),
                'volume': torch.from_numpy(volume).float(),
                'subjid': torch.tensor(subjid, dtype=torch.int64),
                'vol_num': torch.tensor(vol_num, dtype=torch.float64)}

def setup_data_loaders(batch_size=32, shuffle=(True, False, False), train_csv = '', test_csv = ''):
    #set up train dset
    train_dataset = FMRIDataset(csv_file = train_csv, transform = ToTensor())
    #set up shuffle train dset loader --> this is used for training.
    Shuffled_train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                              shuffle=shuffle[0], num_workers=0)
    #set up unshuffled train loader
    #this is used to plot LS -- when we need same dset as in TRAIN, but unshuffled.
    UnShuffled_train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                              shuffle=shuffle[1], num_workers=0)
    #Setup test dset
    test_dataset = FMRIDataset(csv_file = test_csv, transform = ToTensor())
    #set up unshuffled test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, \
                             shuffle=shuffle[2], num_workers=0)
    return {'Shuffled_train':Shuffled_train_loader, 'UnShuffled_train':UnShuffled_train_loader,\
     'test':test_loader}
