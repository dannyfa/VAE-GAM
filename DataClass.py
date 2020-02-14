"""
Script defining fMRIDataset Class and loaders to be used
November 2019
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
           subjid: unique id for each subj. These repeat across vols for same subj.
           age: age for each subject.
           sex: sex for each subject in bin form. Female are coded as 1, males as 0
           volume: np array containing one volume from a given subj
           one_not: a one_hot vector identifying each subj
        """
        unique_subjs = self.df.subjid.unique().tolist()
        subjid = self.df.iloc[idx,1]
        subj_idx = unique_subjs.index(subjid)
        age = self.df.iloc[idx,4]
        sex = self.df.iloc[idx,5]
        task = self.df.iloc[idx,6]
        task_bin = self.df.iloc[idx,7]
        nii = self.df.iloc[idx,3]
        vol_num = self.df.iloc[idx,2]
        fmri = np.array(nib.load(nii).dataobj)
        max = 3284.5 # min is zero! This simplifies norm calc -- becomes x/xmax
        #max = 65536 # used on old dset...
        volume = fmri[:,:,:,vol_num]
        flat_vol = volume.flatten()
        norm_vol = np.true_divide(flat_vol, max).reshape(41,49,35)
        sample = {'subjid': subj_idx, 'volume': norm_vol,
                      'age': age, 'sex': sex, 'task':task, 
                      'subj': subjid, 'task_bin':task_bin}
        if self.transform:
            sample = self.transform(sample)

        return(sample)

class ToTensor(object):
    "Concatenates input array and converts sample arrays to tensors"

    def __call__(self, sample):
        subjid, volume, task = sample['subjid'], sample['volume'], sample['task']
        #Took age & sex covars out
        #concat = np.append(age, sex)
        #concat = np.append (concat, task)
        return{'covariates':torch.tensor(task, dtype=torch.float),
                'volume': torch.from_numpy(volume).float(),
                'subjid': torch.tensor(subjid, dtype=torch.int64)}

def setup_data_loaders(batch_size=32, shuffle=(True, False), csv_file='/home/dfd4/fmri_vae/resampled/preproc_dset.csv'):
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
