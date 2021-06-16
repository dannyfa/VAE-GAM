"""
Script defining fMRIDataset Class and loaders to be used.

"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import nibabel as nib

class FMRIDataset(Dataset):
    def __init__(self, csv_file, transform= None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        """Return the number of samples in dset"""
        return len(self.df)
    def __getitem__(self, idx):
        """Returns a single sample from dset
           Each sample is a dict w/ following keys:
           subjid: unique index for each subj. These repeat across vols for same subj.
           subj: actual string defining a subj identifier
           volume: np array containing one volume from a given subj
           task: task variable (default is binary!)
           trans_x, trans_y, trans_z: translation in x, y, z head axes
           rot_x, rot_y, rot_z: rotation about 3 head axes
        """
        #get subjid and its index
        unique_subjs = self.df.subjid.unique().tolist()
        subj = self.df.iloc[idx,1]
        subj_idx = unique_subjs.index(subj)
        #get all other covariates
        vol_num = self.df.iloc[idx,2]
        nii = self.df.iloc[idx,3]
        #task
        task = self.df.iloc[idx,4]
        #motion params
        trans_x = self.df.iloc[idx,5]
        trans_y = self.df.iloc[idx,6]
        trans_z = self.df.iloc[idx,7]
        rot_x = self.df.iloc[idx,8]
        rot_y = self.df.iloc[idx,9]
        rot_z = self.df.iloc[idx,10]

        fmri = np.array(nib.load(nii).dataobj)
        max = 3284.5
        volume = fmri[:,:,:,vol_num]
        flat_vol = volume.flatten()
        scld_vol = np.true_divide(flat_vol, max).reshape(41,49,35) #scale vol
        #added vol_num to sample
        sample = {'subjid': subj_idx, 'subj': subj, 'volume': scld_vol,
                      'task':task, 'trans_x':trans_x, 'trans_y':trans_y,
                      'trans_z':trans_z, 'rot_x':rot_x, 'rot_y':rot_y,
                      'rot_z':rot_z}
        if self.transform:
            sample = self.transform(sample)

        return(sample)

class ToTensor(object):
    "Converts sample arrays to tensors"
    def __call__(self, sample):
        subjid, volume = sample['subjid'], sample['volume']
        #Concat task w/ mot params by row
        covars = np.array([sample['task'], sample['trans_x'], sample['trans_y'], \
        sample['trans_z'], sample['rot_x'], sample['rot_y'], sample['rot_z']], dtype=np.float64)
        return{'covariates':torch.from_numpy(covars).float(),
                'volume': torch.from_numpy(volume).float(),
                'subjid': torch.tensor(subjid, dtype=torch.int64)}

def setup_data_loaders(batch_size=32, shuffle=(True, False), csv_file=''):
    #Setup the train loaders.
    train_dataset = FMRIDataset(csv_file = csv_file, transform = ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                              shuffle=shuffle[0], num_workers=0)
    # Setup the test loaders.
    test_dataset = FMRIDataset(csv_file = csv_file, transform = ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, \
                             shuffle=shuffle[1], num_workers=0)
    return {'train':train_loader, 'test':test_loader}
