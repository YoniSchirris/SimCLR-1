
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
import json


class TiledTCGADataset(Dataset):
    """Dataset class for tiled WSIs from TCGA
    Requires 'create_complete_data_file.py' to be run in order to get paths + labels"""

    def __init__(self, csv_file, root_dir, transform=None, sampling_strategy='tile', tensor_per_patient=False, 
                    precomputed=False, precomputed_from_run=None, split_num=1, label='msi', split=None, dataset='msi-tcga'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.split_num=1 # Which k-fold for train-val split?
        self.split = split
        self.label=label
        self.dataset = dataset

        # self.labels = pd.read_csv(csv_file)
        self.precomputed=precomputed
        self.precomputed_from_run = precomputed_from_run
        if precomputed:
            if precomputed_from_run:
                self.append_with=f'_{precomputed_from_run}.pt'
            else:
                self.append_with=f'.pt'
        else:
            self.append_with='.jpg'

        self.sampling_strategy=sampling_strategy        # Unused, as we already use root dir + explicit CSV. But used in Splitter.py
        self.tensor_per_patient=tensor_per_patient      # Unused, as we already use root dir + explicit CSV. But used in Splitter.py

        self.csv_file = csv_file
        self.root_dir = root_dir
        if dataset == 'msi-tcga':
            with open('/home/yonis/histogenomics-msc-2019/yoni-code/MsiPrediction/metadata/kather/kather_msi_dot_id_to_tcga_id.json') as f:
                self.dot_id_to_tcga_id = json.load(f)
        self.labels = pd.read_csv(csv_file, converters={'case': str})
        print(f"Successfully loaded labels from {csv_file}, it has {len(self.labels.index)} files.")
        if split:
            if split_num == 0 and split == 'train': # We want the train set, but not a specific fold... so we take all non-test data
                self.labels = self.labels[~(self.labels[f'test']==1)].reset_index() # We train unsupervised on all training data, instead of only specific folds. We don't look at test data, though!
            else:
                if split == 'train' or split == 'val': # We want the train or validation set of a specific fold...
                    append = f'_{split_num}'
                elif split == 'test': # We want all test data
                    append = ''
                self.labels = self.labels[self.labels[f'{split}{append}']==1].reset_index()
            print(f"We use a {split} split of fold {split_num} of {len(self.labels.index)} files.")
            
        if self.tensor_per_patient:
            self.labels = self.labels.groupby('case').mean().reset_index() # We end up with a single row per patient, and the mean of all labels, which is the label
            print(f"Finally, we use a tensor per patient, meaning we now have {len(self.labels.index)} files.")

        self.transform = transform

    def __len__(self):
        full_data_len = len(self.labels.index)
        return full_data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.labels.iloc[idx]

        case_id = str(row['case'])

        if self.label:
            label=row[self.label]
        else:
            label=[]

        # case_id = str(self.labels.at[idx, "case"])
        # dot_id = self.labels.at[idx, "dot_id"]
        # tile_num = self.labels.at[idx, "num"]
        # patient_id = self.dot_id_to_tcga_id[dot_id.split('-')][2]

        # label= self.labels.at[idx, "msi"]

        if not self.tensor_per_patient:
            tile_num = row['num']
            dot_id = row['dot_id']
            img_name = os.path.join(self.root_dir, f'case-{case_id}',
                                    dot_id,
                                    'jpeg',
                                    f'tile{tile_num}{self.append_with}'
                                    )
            
            if self.dataset == 'msi-tcga':
                patient_id = self.dot_id_to_tcga_id[dot_id].split('-')[2]
            elif self.dataset == 'basis':
                patient_id = row['case'].lstrip('case-')
        else:
            img_name = os.path.join(self.root_dir, f'case-{case_id}',
                                    f"pid_{case_id}_tile_vectors_extractor_{self.precomputed_from_run}.pt")
            patient_id = case_id
        if self.precomputed or self.tensor_per_patient:
            tile = torch.load(img_name, map_location='cpu')
        else:
            try:
                tile = io.imread(img_name)
            except ValueError as e:
                print(f"=============== Error: {e}. Corrupt image file: {img_name} ================")
                import sys
                sys.exit()
            if self.transform:
                tile= self.transform(tile)
            else:
                tile= tile.transpose((2, 0, 1))
                tile= torch.from_numpy(tile).float()

        sample = (tile, label, patient_id, img_name)
        return sample
