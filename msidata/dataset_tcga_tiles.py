
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

    def __init__(self, csv_file, root_dir, transform=None, sampling_strategy='tile', tensor_per_patient=False, precomputed=False, precomputed_from_run=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.labels = pd.read_csv(csv_file)
        self.precomputed=precomputed
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
        with open('/home/yonis/histogenomics-msc-2019/yoni-code/MsiPrediction/metadata/kather/kather_msi_dot_id_to_tcga_id.json') as f:
            self.dot_id_to_tcga_id = json.load(f)
        self.labels = pd.read_csv(csv_file, converters={'case': str})
        self.transform = transform

    def __len__(self):
        full_data_len = len(self.labels.index)
        return full_data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.labels.iloc[idx]

        case_id = str(row['case'])
        dot_id = row['dot_id']
        tile_num = row['num']
        patient_id = self.dot_id_to_tcga_id[dot_id].split('-')[2]
        label=row['msi']

        # case_id = str(self.labels.at[idx, "case"])
        # dot_id = self.labels.at[idx, "dot_id"]
        # tile_num = self.labels.at[idx, "num"]
        # patient_id = self.dot_id_to_tcga_id[dot_id.split('-')][2]

        # label= self.labels.at[idx, "msi"]

        img_name = os.path.join(self.root_dir, f'case-{case_id}',
                                dot_id,
                                'jpeg',
                                f'tile{tile_num}{self.append_with}'
                                )

        if self.precomputed:
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
