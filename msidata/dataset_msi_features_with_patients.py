# Written by Yoni Schirris
# June 4 2020

# Script written to load pytorch tensors as input for other models

# This script allows to load tile-level batches and patient-level batches

# This dataloader should be used after save_feature_vectors.py has been run

# This dataloader is necessary since for larger datasets the feature extractor and feature vectors will overflow memory
# Also, the extraction only has to be done once, and then we can test a variety of classification heads rapidly

# The strings returned are a bit messy, as is expected, from here: https://github.com/pytorch/pytorch/issues/6893




from __future__ import print_function, division
import os
import os.path
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import select
import sys
import pprint
import PIL
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random
import time


class PreProcessedMSIFeatureDataset(Dataset):
    """Preprocessed MSI dataset from https://zenodo.org/record/2532612 and https://zenodo.org/record/2530835"""

    def __init__(self, root_dir, transform=None, data_fraction=1, sampling_strategy='tile', device='cpu', balance_classes=False, append_img_path_with=''):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            data_fraction (float): float in [0,1] defining the size of a random subset of all the data used
            sampling_strategy (string): 'tile' or 'patient'. When 'tile', batch_size is taken into account, and batch_size
                tiles are batched. When strategy is 'patient', it will always take all tiles of a single patient. This is
                primarily used for MIL training
        """
        # self.labels = pd.read_csv(csv_file)
        self.balance_classes = balance_classes
        self.sampling_strategy = sampling_strategy
        self.device = 'cpu' # To my knowledge, data loaders generally loads everythig onto CPU. We port to GPU once passed from dataloader
        self.append_img_path_with=append_img_path_with

        if 'msidata' in root_dir:
            # set up stuff for MSI data
            self.label_classes = {'MSS': 0, 'MSIMUT': 1}
            self.task = 'msi'
        elif 'tissuedata' in root_dir:
            # set up stuff for tissue data
            self.label_classes = {'ADIMUC': 0, 'STRMUS': 0, 'TUMSTU': 1}
            self.task = 'cancer'

        self.root_dir = root_dir
        self.setup()

        # Sample per patient, not per tile
        # Add possibility of fixing class imbalance
        self.labels = self._get_labels(data_fraction)
        

        if sampling_strategy == 'patient':
            self.grouped_labels = self.labels.groupby(['patient_id'])
            self.indices_for_groups = list(self.grouped_labels.groups.values())

        self.transform = transform

    def __len__(self, val=False):
        # full_data_len = len([name for label in self.label_classes.keys() for name in os.listdir(f'{self.root_dir}/{label}') if
        #             os.path.isfile(os.path.join(f'{self.root_dir}/{label}', name)) and name.endswith('.png')])
        if self.sampling_strategy == 'tile':
            full_data_len = len(self.labels.index)
        elif self.sampling_strategy == 'patient':
            full_data_len = self.grouped_labels.ngroups
        return full_data_len

    def __getitem__(self, idx):
        t1 = time.time()

        if self.sampling_strategy == 'tile':
            one_or_two_tiles, label, patient_id, img_name = self._get_tile_item(
                idx)
            
        elif self.sampling_strategy == 'patient':
            one_or_two_tiles, label, patient_id, img_name = self._get_patient_items(
                idx)
        # print(f'Loading tile features took {time.time()-t1:.4f} seconds')

        return one_or_two_tiles, label, patient_id, img_name

    def _get_labels(self, data_fraction):

        raw_df = pd.read_csv(self.root_dir + 'data.csv')

        if self.sampling_strategy=='tile':
            # Randomly sample tiles to reduce the amount of data
            subsample_df = raw_df.sample(frac=data_fraction, random_state=42) # NOTE: .sample() shuffles, even when frac=1
        elif self.sampling_strategy=='patient':
            # Randomly sample patients to reduce the amount of data
            # change to sample patients
            #TODO maybe add subsampling from each class separately?
            patients = raw_df['patient_id'].unique()
            subsample_patients = np.random.choice(patients, int(data_fraction*len(patients)))
            subsample_df = raw_df[raw_df['patient_id'].isin(subsample_patients)]
            subsample_df = raw_df.sample(frac=data_fraction, random_state=42) # NOTE: .sample() shuffles, even when frac=1
        else:
            raise NotImplementedError()

        if self.balance_classes:
            msi_patients = subsample_df[subsample_df['label']==1]['patient_id'].unique()
            mss_patients = subsample_df[subsample_df['label']==0]['patient_id'].unique()

            print(f'Pre-downsampling:\nWe have {len(msi_patients)} MSI patients')
            print(f'We have {len(mss_patients)} MSS patients')

            min_class_group_size = min(len(msi_patients), len(mss_patients))

            sub_msi_patients = np.random.choice(msi_patients, min_class_group_size, replace=False)
            sub_mss_patients = np.random.choice(mss_patients, min_class_group_size, replace=False)

            print(f'Post-downsampling:\nWe have {len(sub_msi_patients)} MSI patients')
            print(f'We have {len(sub_mss_patients)} MSS patients')

            sub_patients = np.concatenate((sub_msi_patients, sub_mss_patients), axis=0)

            print(f'Length of concatenated patients: {len(sub_patients)}')
            print(f'Length of unique concatenated patients: {len(set(sub_patients))}')




            balanced_df = subsample_df[subsample_df['patient_id'].isin(sub_patients)]
        else: 
            # well, not actually balanced..
            balanced_df = subsample_df

        print(f"===============\nEffect of subsampling on dataset:\n\nRaw dataset:\n\
            {raw_df.groupby('patient_id').mean().describe()}\
                \n\n\
                Sampled dataset:\
                    \n{balanced_df.groupby('patient_id').mean().describe()}")
        

        return balanced_df

        

    def _get_patient_items(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        indices = self.indices_for_groups[idx] # using iloc, as we pick SOME of the indices

        patient_data = self.labels.loc[indices] # using loc, as we want the original indices, not the current location in the df.

        patient_ids = list(patient_data['patient_id'])

        assert(len(set(patient_ids)) == 1), f'We have more than one patient ID. dataloader is going wrong. patients: {patient_ids}'

        labels = torch.tensor(patient_data['label'].to_numpy(copy=True))

        assert(max(labels) == min(labels)), f'The same patients has different labels somehow: patient: {patient_ids}, labels: {labels}'

        # Loop over relative image paths
        # Replace .png with .pt, this means we can use the same data.csv file
        # Concatenate the root dir and the image relative path
        # torch.load the pickled tensor
        # Place all the tensors in a list
        # Stack the tensors
        vector_paths = [os.path.join(self.root_dir, img.replace('.png', f"{self.append_img_path_with}.pt")) for img in list(patient_data['img'])]
        vectors = torch.stack([torch.load(vector_path, map_location=self.device) for vector_path in vector_paths])

        # import pdb; pdb.set_trace()

        return (vectors, labels, patient_ids, list(patient_data['img']))

    def get_class_distribution(self):
        # take the mean of patient-grouped labels, this gives the patient label as they are all the same
        # group the patients by labels and count the number of patients per label
        # make it a numpy array
        # make it a torch tensor
        # This is 2D, squeeze it to make it 1D as is required by the torch crossentropy class
        # This returns the count for class [0, 1], meaning [MSS, MSIMUT]
        num_classes = torch.from_numpy(self.grouped_labels.mean().groupby('label').count().to_numpy()).squeeze()
        return num_classes.float() / num_classes.sum()


    def _get_tile_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 1]).replace(".png", f"{self.append_img_path_with}.pt") 
        vector = torch.load(img_name, map_location=self.device)
        label = self.labels.iloc[idx, 2]
        patient_id = self.labels.iloc[idx, 3]
        return vector, label, patient_id, img_name

    def setup(self):
        """
        Function that sets up the data.csv files that connect image file name to label
        :param data_dir: the data directory (crc_dx, crc_kr, stad)
        :return: no return
        """
        if not os.path.isfile(os.path.join(self.root_dir, 'data.csv')):
            data = []
            for label in self.label_classes.keys():
                DATAFILENAME = f'data.csv'
                DIR = self.root_dir
                SUBDIR = os.path.join(DIR, label)

                for name in os.listdir(SUBDIR):
                    if name.endswith('.png'):

                        if self.task == 'msi':
                                # 1st column of labels holds LABEL/IMAGE_NAME.png
                                # IMAGE_NAME is blk-ABCDEGHIJKLMNOP-TCGA-AA-####-01Z-00-DX1.png
                                # where #### is the patient ID
                            patient_id = name.split('-')[4]
                        elif self.task == 'cancer':
                            patient_id = name.split('-')[1].split('.')[0]
                        else:
                            raise NotImplementedError

                        label_class = self.label_classes[label]
                        data.append([f'{label}/{name}', label_class, patient_id])
            df = pd.DataFrame(data=data, columns=['img', 'label', 'patient_id'])
            df.to_csv(DIR + DATAFILENAME)
            return
        else:
            return
