# Written by Yoni Schirris
# Wed 3 June 2020



# Script that transforms each tile given by a "regular" dataloader to a .pt feature vector with the same name
# This makes it easy to adjust existing dataloaders to load these vectors instead of images

# This script should be run after having trained a SimCLR feature extractor with main.py

# After this script, one can use the tile- or patient-level vector dataloader to test classification heads

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

import pandas as pd
import time
import datetime



def infer_and_save(loader, context_model, device):
    for step, (x, y, patient, img_names) in enumerate(loader):
        x = x.to(device)

        print(f'Size of x: {x.shape}')

        # get encoding
        with torch.no_grad():
            h, z = context_model(x)

        h = h.detach()

        print(f'Size of h: {h.shape}')

        for i, img_name in enumerate(img_names):
            feature_vec = h[i]
            torch.save(feature_vec, img_name.replace('.png', '.pt'))

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")


def save_features(context_model, train_loader, test_loader, device):
    infer_and_save(train_loader, context_model, device)
    infer_and_save(test_loader, context_model, device)
    


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the dataset. sample_strategy is patient, meaning we get all tiles for a patient in a single _get
    train_dataset = dataset_msi(
            root_dir=args.path_to_msi_data, 
            transform=TransformsSimCLR(size=224).test_transform, 
            data_fraction=args.data_create_feature_vectors_fraction
            )

    test_dataset = dataset_msi(
            root_dir=args.path_to_test_msi_data, 
            transform=TransformsSimCLR(size=224).test_transform, 
            data_fraction=args.data_create_feature_vectors_fraction
            )


    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
        )

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
        )

    simclr_model, _, _ = load_model(args, train_loader, reload_model=True)
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    save_features(simclr_model, train_loader, test_loader, args.device)


    
