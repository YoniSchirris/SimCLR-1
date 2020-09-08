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
import json

from experiment import ex
from model import load_model
from utils import post_config_hook

from modules import LogisticRegression
from modules.transformations import TransformsSimCLR

from msidata.dataset_msi import PreProcessedMSIDataset as dataset_msi
from msidata.dataset_msi import PreProcessedMSIDataset
from msidata.dataset_tcga_tiles import TiledTCGADataset
from msidata.dataset_tcga_tiles import TiledTCGADataset as dataset_tcga



import os

import pandas as pd
import time
import datetime


def infer_and_save(loader, context_model, device, append_with='', model_type=None):
    if isinstance(loader.dataset, TiledTCGADataset):
        extension = '.jpg'
    elif isinstance(loader.dataset, PreProcessedMSIDataset):
        extension = '.png'
    else:
        raise NotImplementedError
    for step, (x, y, patient, img_names) in enumerate(loader):
        print(f"[ Step {step} / {len(loader)} ]")
        x = x.to(device)
        # get encoding
        if not model_type or model_type == 'simclr':
            # if there's any location calling this function without model_type, keep standard behavior, expecting simclr model
            with torch.no_grad():
                h, z = context_model(x)
        else:
            if model_type in ['imagenet-resnet18', 'imagenet-resnet50', 'imagenet-shufflenet-v1_x1_0', 'byol']:
                context_model.fc = torch.nn.Identity()
                with torch.no_grad():
                    h = context_model(x)
            else:
                raise NotImplementedError

        h = h.detach()

        for i, img_name in enumerate(img_names):
            tensor_name = img_name.replace(extension, f'{append_with}.pt')
            assert(
                tensor_name != img_name), f"You are about to overwrite the original image with the tensor. You don't want this. Extension is set to {extension}, while the filename is {img_name}"
            torch.save(h[i].clone(), tensor_name)

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")


def aggregate_patient_vectors(args, root_dir, append_with='', grid=False):
    print(f"## Aggregating vectors per patient in {root_dir}")
    if args.dataset == "msi-kather":
        data = pd.read_csv(os.path.join(root_dir, 'data.csv'))
        identifier = 'patient_id' # We don't have clear information about separate WSIs, so we stack all tiles from a patient in a single tensor
        extension = '.png'
    elif args.dataset == "msi-tcga":
        data = pd.read_csv(root_dir)  # csv is given as root dir
        identifier = 'dot_id'   # We have a clear dot_id per WSI, and will thus create a stack / grid for each WSI separately
        extension = '.jpg'

        # Creating ABSOLUTE image paths here
        data['img'] = data.apply(lambda x: os.path.join(args.root_dir_for_tcga_tiles, f"case-{x['case']}",
                                                        x['dot_id'],
                                                        'jpeg',
                                                        f"tile{x['num']}{extension}"
                                                        ), axis=1)

    for idx, idd in enumerate(data[identifier].unique()):    # id here 
        relative_img_paths = data[data[identifier] == idd]['img']

        relative_tensor_paths = [img_path.replace(extension, f'{append_with}.pt') for img_path in relative_img_paths]

        if not grid:
            # We simply stack them. This is useful for DeepMIL
            vectors = torch.stack([torch.load(tensor_path, map_location='cpu')
                                   for tensor_path in relative_tensor_paths])
        else:
            # We use the information from our preprocessor jsons to place the feature vectors in a grid
            # TODO Shouldn't we do this ONCE and place it in the data .csv file? This would make it much much more efficient.
            # TODO Reading 3 million json files will take a terribly long time
            # TODO Run this on surfsara to see if it works...

            num_features = torch.load(
                relative_tensor_paths[0], map_location='cpu').shape[0]

            relative_json_paths = [img_path.replace('.jpg', '.json').replace(
                '/jpeg/', '/json/') for img_path in relative_img_paths]
            coords = []
            for absolute_json_path in relative_json_paths:
                with open(os.path.join(absolute_json_path), 'r') as f:
                    tile_data = json.load(f)
                    coords.append([tile_data['row_idx'], tile_data['col_idx']])

            coords = np.array(coords)

            img_paths_and_coords = (relative_img_paths, coords)

            shape_for_patient_grid = list(np.amax(coords, axis=0) + 1) # If max index if 80, we want a tensor of size 81
            shape_for_patient_grid.append(num_features)

            patient_grid = torch.zeros(tuple(shape_for_patient_grid))
            for tensor_path, tile_coords in zip(relative_tensor_paths, coords):
                patient_grid[tuple(tile_coords)] = torch.load(
                    tensor_path, map_location='cpu')

            # Whether it's a grid or not, we call it "vectors" to save it later
            vectors = patient_grid

        if args.dataset == 'msi-kather':
            relative_img_paths_dirs = [
                '/'.join(path.split('/')[:-1]) for path in relative_img_paths]
            relative_dir = set(relative_img_paths_dirs)
            assert len(
                relative_dir) == 1, f"A single patient have several labels! see {relative_img_paths}"
        elif args.dataset == 'msi-tcga':
            # saving them in the case dir, not the dot_id dir
            relative_img_paths_dirs = [
                '/'.join(path.split('/')[:-3]) for path in relative_img_paths]
            relative_dir = set(relative_img_paths_dirs)
            assert len(
                relative_dir) == 1, f"A single patient presumably has different case ids ! see {relative_img_paths}"

        # Fine, since all are the same
        relative_dir = relative_img_paths_dirs[0]

        # Note that the root dir is a directory for msi-kather, but a .csv for msi-tcga
        # However, os.path.join removes the .csv, as it does not make sense. This is why it actually works.
        if grid:
            # args.path_to_msi_data has `subsample_n` in its name if we have a subsample
            # since we might save a grid for ALL tiles and a subsapmle, we require a different naming for both
            # so we check if the path to msi data has 'subsample' in it, get 'n', and add this to the 
            if 'subsample' in args.path_to_msi_data:
                path_to_data = args.path_to_msi_data.split('_')
                n = path_to_data[path_to_data.index('subsample')+1].rstrip('.csv')
                append_with_ = append_with + f"_subsample_{n}"
            else:
                append_with_ = append_with

            filename = os.path.join(
                root_dir, relative_dir, f'pid_{idd}_tile_grid_extractor{append_with_}.pt')
            paths_filename = os.path.join(
                root_dir, relative_dir, f"pid_{idd}_tile_grid_paths_and_coords_extractor{append_with_}.pt")
            torch.save(img_paths_and_coords, paths_filename)
        else:
            filename = os.path.join(
                root_dir, relative_dir, f'pid_{idd}_tile_vectors_extractor{append_with_}.pt')
            paths_filename = os.path.join(
                root_dir, relative_dir, f"pid_{idd}_tile_vector_paths_extractor{append_with_}.pt")
            torch.save(relative_img_paths, paths_filename)

        torch.save(vectors, filename)
        print(f'[ {datetime.datetime.now()} ] \t [ {idx} / {len(data[identifier].unique())} ] \t Saving {filename}')


def save_features(context_model, train_loader, test_loader, val_loader, device, append_with=''):
    infer_and_save(train_loader, context_model, device, append_with)
    infer_and_save(test_loader, context_model, device, append_with)
    infer_and_save(val_loader, context_model, device, append_with)


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)

    assert args.data_create_feature_vectors_fraction == 1, "Only works if we transform all tiles to vectors"

    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    if 'create_feature_grid' not in vars(args).keys():
        print("### create_feature_grid not found in config, we will create a grid, then!")
        args.create_feature_grid = True

    if not args.use_precomputed_features:
        # model path is generally e.g. logs/pretrain/51
        run_id = args.model_path.split('/')[-1]

        # Load the dataset. sample_strategy is patient, meaning we get all tiles for a patient in a single _get
        if args.dataset == "msi-kather":
            train_dataset = dataset_msi(
                root_dir=args.path_to_msi_data,
                transform=TransformsSimCLR(size=224).test_transform,
                data_fraction=args.data_testing_train_fraction,
                seed=args.seed,
                label=label,
                load_labels_from_run=load_labels_from_run
            )
            test_dataset = dataset_msi(
                root_dir=args.path_to_test_msi_data,
                transform=TransformsSimCLR(size=224).test_transform,
                data_fraction=args.data_testing_test_fraction,
                seed=args.seed,
                label=label,
                load_labels_from_run=load_labels_from_run
            )
        elif args.dataset == "msi-tcga":
            args.data_pretrain_fraction = 1
            assert (
                '.csv' in args.path_to_msi_data), "Please provide the tcga .csv file in path_to_msi_data"
            assert ('root_dir_for_tcga_tiles' in vars(args).keys()
                    ), "Please provide the root dir for the tcga tiles"
            train_dataset = dataset_tcga(
                csv_file=args.path_to_msi_data,
                root_dir=args.root_dir_for_tcga_tiles,
                transform=TransformsSimCLR(size=224).test_transform,
                split_num=args.kfold,
                label=None,
                split='train'
            )
            test_dataset = dataset_tcga(
                csv_file=args.path_to_msi_data,
                root_dir=args.root_dir_for_tcga_tiles,
                transform=TransformsSimCLR(size=224).test_transform,
                split_num=args.kfold,
                label=None,
                split='test'
            )
            val_dataset = dataset_tcga(
                csv_file=args.path_to_msi_data,
                root_dir=args.root_dir_for_tcga_tiles,
                transform=TransformsSimCLR(size=224).test_transform,
                split_num=args.kfold,
                label=None,
                split='val'
            )
        else:
            raise NotImplementedError

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

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
        )

        simclr_model, _, _ = load_model(args, train_loader, reload_model=True)
        simclr_model = simclr_model.to(args.device)
        simclr_model.eval()

        save_features(simclr_model, train_loader, test_loader,
                      val_loader, args.device, append_with=f'_{run_id}')

    if args.use_precomputed_features:
        run_id = args.use_precomputed_features_id

    aggregate_patient_vectors(args, root_dir=args.path_to_msi_data,
                              append_with=f'_{run_id}', grid=args.create_feature_grid)
