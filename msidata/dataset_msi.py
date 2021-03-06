
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

    def __init__(self, root_dir, transform=None, data_fraction=1, seed=42, label='label', load_labels_from_run=''):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.labels = pd.read_csv(csv_file)

        self.label = label
        self.transform = transform
        self.root_dir = root_dir

        if 'msidata' in root_dir:
            # set up stuff for MSI data
            self.label_classes = {'MSS': 0, 'MSIMUT': 1}
            self.task = 'msi'
        elif 'tissuedata' in root_dir:
            # set up stuff for tissue data
            self.label_classes = {'ADIMUC': 0, 'STRMUS': 0, 'TUMSTU': 1}
            self.task = 'cancer'
        
        self.setup()
        path_to_dataset = self.root_dir + ('data.csv' if not load_labels_from_run else f'data_{load_labels_from_run}.csv')
        print(f"Loading data from {path_to_dataset}")
        self.labels = pd.read_csv(path_to_dataset).sample(frac=data_fraction, random_state=seed).reset_index(drop=True)

        assert(self.label in self.labels.columns), f"The requested label: {self.label} is not available in the current dataset from {path_to_dataset}. The dataset has the following columns: {list(self.labels.columns)}"

    def __len__(self, val=False):
        # full_data_len = len([name for label in self.label_classes.keys() for name in os.listdir(f'{self.root_dir}/{label}') if
        #             os.path.isfile(os.path.join(f'{self.root_dir}/{label}', name)) and name.endswith('.png')])
        full_data_len = len(self.labels.index)
        return full_data_len

    def __getitem__(self, idx):
        t1 = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        t2=time.time()

        row = self.labels.iloc[idx]
        label = row[self.label]
        im_rel_path = row[1]

        if self.task == 'msi':
            # 1st column of labels holds LABEL/IMAGE_NAME.png
            # IMAGE_NAME is blk-ABCDEGHIJKLMNOP-TCGA-AA-####-01Z-00-DX1.png
            # where #### is the patient ID
            patient_id = row[1].split('/')[1].split('-')[4]
        elif self.task == 'cancer':
            patient_id = row[1].split('-')[1].split('.')[0]
            # TODO check the patient id setup. No idea what to do with it now.
        else:
            patient_id = []

        t3=time.time()

        img_name = os.path.join(self.root_dir, im_rel_path)
        tile = PIL.Image.open(img_name) # ToPILImage takes either torch tensor or ndarray

        t4=time.time()
        
        #  ---- Transform image to torch with right dimensions
        # imt = image.transpose((2, 0, 1)) # THIS IS ALREADY HANDLED IN TOTENSOR
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # ----- End of transform

        #tile = torch.from_numpy(imt).float()  

        if self.transform:
            tile = self.transform(tile)
        else:
            tile = tile.transpose((2,0,1))
            tile = torch.from_numpy(tile).float()

        t5=time.time()

        total_time=t5-t1
        print_time=False
        if print_time:
            print(f'Total: {total_time} \t tolist: {(t2-t1)/total_time} \t get_info: {(t3-t2)/total_time} \t load_im: {(t4-t2)/total_time} \t transform_im {(t5-t4)/total_time}')

        return tile, label, hash(patient_id), img_name, []

    def get_class_balance(self):
    
        label_mean = self.labels['label'].mean()
        label_weights = [1, (1-label_mean)/label_mean]
        print(f"==== Setting class balancing weights for {self.task}: {label_weights} =====")
        return torch.tensor(label_weights)

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
