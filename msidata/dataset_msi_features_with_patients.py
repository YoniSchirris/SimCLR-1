# Written by Yoni Schirris
# June 4 2020

# Script written to load pytorch tensors as input for other models

# This script allows to load tile-level batches and patient-level batches

# This dataloader should be used after save_feature_vectors.py has been run

# This dataloader is necessary since for larger datasets the feature extractor and feature vectors will overflow memory
# Also, the extraction only has to be done once, and then we can test a variety of classification heads rapidly

# 




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

    def __init__(self, root_dir, transform=None, data_fraction=1, sample_strategy='tile'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            data_fraction (float): float in [0,1] defining the size of a random subset of all the data used
            sample_strategy (string): 'tile' or 'patient'. When 'tile', batch_size is taken into account, and batch_size
                tiles are batched. When strategy is 'patient', it will always take all tiles of a single patient. This is
                primarily used for MIL training
        """
        # self.labels = pd.read_csv(csv_file)

        self.sample_strategy = sample_strategy

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
        self.labels = pd.read_csv(
            self.root_dir + 'data.csv').sample(frac=data_fraction, random_state=42)

        if sample_strategy == 'patient':
            self.grouped_labels = self.labels.groupby(['patient_id'])
            self.indices_for_groups = list(self.grouped_labels.groups.values())

        self.transform = transform

    def __len__(self, val=False):
        # full_data_len = len([name for label in self.label_classes.keys() for name in os.listdir(f'{self.root_dir}/{label}') if
        #             os.path.isfile(os.path.join(f'{self.root_dir}/{label}', name)) and name.endswith('.png')])
        if self.sample_strategy == 'tile':
            full_data_len = len(self.labels.index)
        elif self.sample_strategy == 'patient':
            full_data_len = self.grouped_labels.ngroups
        return full_data_len

    def __getitem__(self, idx):

        if self.sample_strategy == 'tile':
            one_or_two_tiles, label, patient_id, img_name = self._get_tile_item(
                idx)
            
        elif self.sample_strategy == 'patient':
            one_or_two_tiles, label, patient_id, img_name = self._get_patient_items(
                idx)

        return one_or_two_tiles, label, patient_id, img_name

    def _get_patient_items(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        indices = self.indices_for_groups[idx]

        patient_data = self.labels.iloc[indices]

        patient_ids = list(patient_data['patient_id'])

        labels = list(patient_data['label'])

        # Loop over relative image paths
        # Replace .png with .pt, this means we can use the same data.csv file
        # Concatenate the root dir and the image relative path
        # torch.load the pickled tensor
        # Place all the tensors in a list
        # Stack the tensors
        vector_paths = [os.path.join(self.root_dir, img.replace('.png', '.pt')) for img in list(patient_data['img'])]
        vectors = torch.stack([torch.load(vector_path) for vector_path in vector_paths])

        return vectors, labels, patient_ids, list(patient_data['img'])


    def _get_tile_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 1]).replace(".png", ".pt")
        vector = torch.load(img_name)
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
                            print(name)
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
