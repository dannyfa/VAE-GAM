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
        self.df = pd.read_csv(csv_file) # see if good to specify dtypes here
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
        maxsig = 65536 # see if this is still same value?
        #create ID mat
        # Each row is  a one-hot for each unique subjid in df
        id_mat = np.identity(len(unique_subjs))
        subjid = self.df.iloc[idx,1]
        onehot_idx = unique_subjs.index(subjid)
        one_hot = id_mat[onehot_idx]
        age = self.df.iloc[idx,4]
        sex = self.df.iloc[idx,5]
        task = self.df.iloc[idx,6]
        nii = self.df.iloc[idx,3]
        vol_num = self.df.iloc[idx,2]
        fmri = np.array(nib.load(nii).dataobj)
        fmri_norm = np.true_divide(fmri, maxsig) # check if ok
        volume = fmri_norm[:,:,:,vol_num]
        sample = {'subjid': subjid, 'volume': volume, 'one_hot': one_hot,
                      'age': age, 'sex': sex, "task":task}
        if self.transform:
            sample = self.transform(sample)

        return(sample)

class ToTensor(object):
    "Concatenates input array and converts sample arrays to tensors"
    
    def __call__(self, sample):
        subjid, volume, id_vector, age, sex, task = sample['subjid'], sample['volume'], sample['one_hot'], sample['age'], sample['sex'], sample['task']
        concat = np.append(id_vector, age)
        concat = np.append (concat, sex)
        concat = np.append (concat, task)
        return{'x_in':torch.from_numpy(concat),
                'volume': torch.from_numpy(volume),
                'subjid': subjid}

def setup_data_loaders(batch_size=32, shuffle=(True, False), csv_file='/home/dfd4/fmri_vae/preproc_dset.csv'):
    # Setup the train loaders.
    #Set up num workers to zero for run time error 
    train_dataset = FMRIDataset(csv_file = csv_file, transform = ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                              shuffle=shuffle[0], num_workers=0)
    # Setup the test loaders.
    test_dataset = FMRIDataset(csv_file = csv_file, transform = ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, \
                             shuffle=shuffle[1], num_workers=0)
    return {'train':train_loader, 'test':test_loader}
