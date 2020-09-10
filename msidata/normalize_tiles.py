import os
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from msidata.dataset_msi import PreProcessedMSIDataset as dataset_msi
from msidata.save_feature_vectors import infer_and_save, aggregate_patient_vectors
from msidata.dataset_msi_features_with_patients import PreProcessedMSIFeatureDataset
from msidata.dataset_tcga_tiles import TiledTCGADataset as dataset_tcga

import pandas as pd
import time
import datetime
import os
import json

from sklearn import metrics


def main():

    # Set arguments
    workers=6
    path_to_msi_data = '/project/yonis/brca_dx/tiled_data_large/'
    root_dir_for_tcga_tiles = '/home/yonis/histogenomics-msc-2019/yoni-code/MsiPrediction/metadata/tcga/brca_dx/data_tcga_brca_dxwith_ddr_labels_and_splits.csv'
    kfold = 1
    ddr_label=None
    batch_size=128
    henorm='macenko'
    path_to_target_im='/project/yonis/tcga_brca_dx/target_image_macenko/Ref.png'


    # Build transformation
    transformation = torchvision.transforms.Compose(
            [
                MyHETransform(henorm=henorm, path_to_target_im=path_to_target_im),
                torchvision.transforms.ToTensor()
            ]
    )
    # Get datasets
    train_dataset, val_dataset, test_dataset = [dataset_tcga(
        csv_file=path_to_msi_data, 
        root_dir=root_dir_for_tcga_tiles, 
        transform=transformation,
        split_num=kfold,
        label=ddr_label,
        split=split
    ) for split in ['train', 'val', 'test']]

    # Get dataloader
    train_loader, val_loader, test_loader, = [torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers
    ) for dataset in [train_dataset, val_dataset, test_dataset]]

    # Loop over data
    for loader in [train_loader, val_loader, test_loader]:
        for step, data in enumerate(loader):
            # Save images to disk
            print(f"[{datetime.datetime.now()} | Step {step} / {len(loader)}]")
            tiles, labels, patient_ids, img_names = data
             for tile, label, patient_id, img_name in zip(tiles, labels, patient_ids, img_names):
                 # Save each tile separately
                 tile = torchvision.transforms.functional.to_pil_image(tile)
                 new_img_name = img_name.replace('.jpg', '_norm_mac_kath.jpg')
                 new_absolute_tile_path = os.path.join(root_dir, new_img_name)
                #  assert(not os.path.isfile(new_absolute_tile_path)), f"{new_absolute_tile_path} already exists..."
                if os.path.isfile(new_absolute_tile_path):
                    print(f"{new_absolute_tile_path} already exists! Overwriting for now!")
                 tile.save(new_absolute_tile_path)

          

if __name__=="__main__":
    main()