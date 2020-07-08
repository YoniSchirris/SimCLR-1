
from __future__ import print_function, division
import os, os.path
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

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.labels = pd.read_csv(csv_file)

        self.csv_file = csv_file
        self.root_dir = root_dir
        with open('/home/yonis/histogenomics-msc-2019/yoni-code/MsiPrediction/metadata/kather/kather_msi_dot_id_to_tcga_id.json') as f:
            self.dot_id_to_tcga_id =  json.load(f)
        self.labels = pd.read_csv(csv_file, converters={'case': str})
        self.transform = transform

    def __len__(self):
        full_data_len = len(self.labels.index)
        return full_data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        case_id = str(self.labels.at[idx, "case"])
        dot_id = self.labels.at[idx, "dot_id"]
        tile_num = self.labels.at[idx, "num"]
        img_name = os.path.join(self.root_dir, f'case{case_id}',
                                dot_id,
                                'jpeg',
                                f'tile{tile_num}.jpg'
                                )
        image = io.imread(img_name)
        patient_id = self.dot_id_to_tcga_id[dot_id.split('-')[2]

        label = self.labels.at[idx, "msi"]

        #  ---- Transform image to torch with right dimensions
        im = np.asarray(image)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        imt = im.transpose((2, 0, 1))
        # ----- End of transform

        tile = torch.from_numpy(imt).float()

        if self.transform:
            tile = self.transform(tile)
        sample = (tile, patient_id, label, img_name)
        return sample