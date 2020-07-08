
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


class PreProcessedMSIDataset(Dataset):
    """Preprocessed MSI dataset from https://zenodo.org/record/2532612 and https://zenodo.org/record/2530835"""

    def __init__(self, root_dir, transform=None, data_fraction=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.labels = pd.read_csv(csv_file)

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
        self.labels = pd.read_csv(self.root_dir + 'data.csv').sample(frac=data_fraction, random_state=42)
        self.transform = transform

    def __len__(self, val=False):
        # full_data_len = len([name for label in self.label_classes.keys() for name in os.listdir(f'{self.root_dir}/{label}') if
        #             os.path.isfile(os.path.join(f'{self.root_dir}/{label}', name)) and name.endswith('.png')])
        full_data_len = len(self.labels.index)
        return full_data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 1])
        image = io.imread(img_name)
        label = self.labels.iloc[idx, 2]

        #  ---- Transform image to torch with right dimensions
        im = np.asarray(image)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        #TODO Check this color axis swap, this doesn't seem to make sense and

        imt = im.transpose((2, 0, 1))
        # ----- End of transform

        tile = torch.from_numpy(imt).float()

        if self.task == 'msi':
            # 1st column of labels holds LABEL/IMAGE_NAME.png
            # IMAGE_NAME is blk-ABCDEGHIJKLMNOP-TCGA-AA-####-01Z-00-DX1.png
            # where #### is the patient ID
            patient_id = self.labels.iloc[idx,1].split('/')[1].split('-')[4]
        elif self.task == 'cancer':
            patient_id = self.labels.iloc[idx,1].split('-')[1].split('.')[0]
            # TODO check the patient id setup. No idea what to do with it now.
        else:
            patient_id = []

        if self.transform:
            tile = self.transform(tile)

        return tile, label, hash(patient_id), img_name

    def setup(self):
        """
        Function that sets up the data.csv files that connect image file name to label
        :param data_dir: the data directory (crc_dx, crc_kr, stad)
        :return: no return
        """
        if not os.path.isfile(self.root_dir + 'data.csv'):
            data = []
            for label in self.label_classes.keys():
                DATAFILENAME = f'data.csv'
                DIR = self.root_dir
                SUBDIR = f'{DIR}{label}/'

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
            df = pd.read_csv(os.path.join(self.root_dir, 'data.csv'))
            cols = df.columns
            assert 'img' in cols and 'label' in cols and 'patient_id' in cols, f"data.csv seems to be corrupt. Columns are {cols}. Might have to remove the data.csv and rerun."
            return
