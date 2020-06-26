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

import os

import pandas as pd
import time
import datetime



def infer_and_save(loader, context_model, device, append_with='', model_type=None):
    for step, (x, y, patient, img_names) in enumerate(loader):
        x = x.to(device)
        # get encoding
        if not model_type or model_type=='simclr':
            # if there's any location calling this function without model_type, keep standard behavior, expecting simclr model
            with torch.no_grad():
                h, z = context_model(x)
        else:
            if model_type in ['imagenet-resnet18', 'imagenet-resnet50', 'imagenet-shufflenet-v1_x1_0':]
                context_model.fc = torch.nn.Identity()
                with torch.no_grad():
                    h = context_model(x)
            else:
                raise NotImplementedError

        h = h.detach()

        for i, img_name in enumerate(img_names):
            feature_vec = h[i]
            torch.save(feature_vec, img_name.replace('.png', f'{append_with}.pt'))

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")


def aggregate_patient_vectors(root_dir, append_with=''):
    print(f"## Aggregating vectors per patient in {root_dir}")
    data = pd.read_csv(os.path.join(root_dir, 'data.csv'))

    for patient_id in data['patient_id'].unique():
        relative_img_paths = data[data['patient_id']==patient_id]['img']
        vectors = torch.stack([torch.load(os.path.join(root_dir, vector_path.replace('.png',f'{append_with}.pt')), map_location='cpu') for vector_path in relative_img_paths])

        relative_img_paths_dirs = ['/'.join(path.split('/')[:-1]) for path in relative_img_paths]

        relative_dir = set(relative_img_paths_dirs)

        assert len(relative_dir)==1, f"A single patient have several labels! see {relative_img_paths}"

        relative_dir = relative_img_paths_dirs[0] # Fine, since all are the same

        filename = os.path.join(root_dir, relative_dir, f'pid_{patient_id}_tile_vectors_extractor{append_with}.pt')
        torch.save(vectors, filename)
        print(f'Saving {filename}')


def save_features(context_model, train_loader, test_loader, device, append_with=''):
    infer_and_save(train_loader, context_model, device, append_with)
    infer_and_save(test_loader, context_model, device, append_with)
    

 

@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)

    assert args.data_create_feature_vectors_fraction == 1, "Only works if we transform all tiles to vectors"

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

    run_id = args.model_path.split('/')[-1] # model path is generally e.g. logs/pretrain/51

    simclr_model, _, _ = load_model(args, train_loader, reload_model=True)
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    save_features(simclr_model, train_loader, test_loader, args.device, append_with=f'_{run_id}')

    aggregate_patient_vectors(root_dir=args.path_to_msi_data, append_with=f'_{run_id}')
    aggregate_patient_vectors(root_dir=args.path_to_test_msi_data, append_with=f'_{run_id}')


    
