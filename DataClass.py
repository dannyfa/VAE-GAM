"""
Script defining fMRIDataset Class and loaders to be used
Modf March 2020
-- Added convolved task var and motion vars

ToDos
Add proper train/test split and better random shuffling here 
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import nibabel as nib

class FMRIDataset(Dataset):
    """This is a slightly different version from Rachel's original """
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
           age: age for each subject.
           sex: sex for each subject in bin form. Female are coded as 1, males as 0
           volume: np array containing one volume from a given subj
           task: real-valued task  -- after HRF convolved
           task_bin: binary task var (needed for other things)
           x_mot, y_mot, z_mot: translation in x, y, z axis respectively
           pitch, roll, yaw: rotation across 3 axis (same as canonical defs for fMRI)
        """
        #get subjid and its index
        unique_subjs = self.df.subjid.unique().tolist()
        subjid = self.df.iloc[idx,1]
        subj_idx = unique_subjs.index(subjid)
        #get all other covariates
        #vol # and nii path
        vol_num = self.df.iloc[idx,2]
        nii = self.df.iloc[idx,3]
        #age, sex
        age = self.df.iloc[idx,4]
        sex = self.df.iloc[idx,5]
        #convolved and bin task_bin
        task = self.df.iloc[idx,6]
        task_bin = self.df.iloc[idx,7]
        #motion params
        x_mot = self.df.iloc[idx,8]
        y_mot = self.df.iloc[idx,9]
        z_mot = self.df.iloc[idx,10]
        pitch_mot = self.df.iloc[idx,11]
        roll_mot = self.df.iloc[idx,12]
        yaw_mot = self.df.iloc[idx,13]

        fmri = np.array(nib.load(nii).dataobj)
        max = 3284.5 # min is zero! This simplifies norm calc -- becomes x/xmax
        #max = 65536 # used on old dset...
        volume = fmri[:,:,:,vol_num]
        flat_vol = volume.flatten()
        norm_vol = np.true_divide(flat_vol, max).reshape(41,49,35)
        sample = {'subjid': subj_idx, 'volume': norm_vol,
                      'age': age, 'sex': sex, 'task':task,
                      'subj': subjid, 'task_bin':task_bin, 'x':x_mot,
                      'y':y_mot, 'z':z_mot, 'pitch':pitch_mot, 'roll':roll_mot,
                      'yaw':yaw_mot}
        if self.transform:
            sample = self.transform(sample)

        return(sample)

class ToTensor(object):
    "Converts sample arrays to tensors"

    def __call__(self, sample):
        subjid, volume = sample['subjid'], sample['volume']
        #Concat task w/ mot params by row
        covars = np.array([sample['task'], sample['x'], sample['y'], sample['z'], \
        sample['pitch'], sample['roll'], sample['yaw']], dtype=np.float64)
        return{'covariates':torch.from_numpy(covars).float(),
                'volume': torch.from_numpy(volume).float(),
                'subjid': torch.tensor(subjid, dtype=torch.int64)}

def setup_data_loaders(batch_size=32, shuffle=(True, False), csv_file=''):
    #Set num workers to zero to avoid runtime error msg.
    #This might need further looking into when we use larger dsets.
    #Setup the train loaders.
    train_dataset = FMRIDataset(csv_file = csv_file, transform = ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                              shuffle=shuffle[0], num_workers=0)
    # Setup the test loaders.
    test_dataset = FMRIDataset(csv_file = csv_file, transform = ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, \
                             shuffle=shuffle[1], num_workers=0)
    return {'train':train_loader, 'test':test_loader}
