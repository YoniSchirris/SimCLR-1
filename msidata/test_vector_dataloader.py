# Written by Yoni Schirris
# Wed 4 June 2020

# Testing the dataloader for pytorch vectors


import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np

from experiment import ex
from model import load_model
from utils import post_config_hook

from modules import LogisticRegression
from modules.transformations import TransformsSimCLR

from msidata.dataset_msi import PreProcessedMSIDataset as dataset_msi
from msidata.wip_patient_dataset_msi import PreProcessedMSIFeatureDataset as feature_dataset_msi

import pandas as pd
import time
import datetime

@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the dataset. sample_strategy is patient, meaning we get all tiles for a patient in a single _get
    train_tile_dataset = feature_dataset_msi(
            root_dir=args.path_to_msi_data, 
            sample_strategy='tile',
            data_fraction=1
            )

    train_patient_dataset = feature_dataset_msi(
            root_dir=args.path_to_msi_data,  
            sample_strategy='patient',
            data_fraction=1
            )

    tile_vector_loader = torch.utils.data.DataLoader(
            train_tile_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
        )

    patient_loader = torch.utils.data.DataLoader(
            train_patient_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
        )

    for data in tile_vector_loader:
        # We expect this to be of shape batch_size x channels
        print('tiles..')
        random_tiles = data[0]
        print(f'shape of tiles: {random_tiles.shape}')
    
    for data in patient_loader:
        # We expect this to be of shape 1 x <variable> x channels
        print('patients..')
        patients_tiles = data[0]
        print(f'shape of patients tiles: {patients_tiles.shape}')

    


    
    print("SUCCESS")