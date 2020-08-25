
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

    def __init__(self, csv_file, root_dir, transform=None, sampling_strategy='tile', tensor_per_patient=False, tensor_per_wsi=False, load_tensor_grid=False,
                    precomputed=False, precomputed_from_run=None, split_num=1, label='msi', split=None, dataset='msi-tcga', stack_grid=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        assert (not tensor_per_patient), "tensor_per_patient is deprecated, please use tensor_per_wsi, as this makes more sense generally, especially for grids"
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

        self.sampling_strategy=sampling_strategy    # Unused, as we already use root dir + explicit CSV. But used in Splitter.py
        self.tensor_per_wsi=tensor_per_wsi          # Unused, as we already use root dir + explicit CSV. But used in Splitter.py
        self.load_tensor_grid = load_tensor_grid    # Append filepath with specifics to load a grid of feature tensors
        self.stack_grid = stack_grid                # Load a grid, remove spatial structure, remove 0-tensors, and pad it up

        self.csv_file = csv_file
        self.root_dir = root_dir
        if dataset == 'msi-tcga':
            with open('/home/yonis/histogenomics-msc-2019/yoni-code/MsiPrediction/metadata/tcga/tcga_crc_and_brca_dot_id_to_tcga_id.json') as f:
                self.dot_id_to_tcga_id = json.load(f)
        self.labels = pd.read_csv(csv_file, converters={'case': str})

        if self.label:
            original_tiles_with_labels = len(self.labels)
            self.labels = self.labels.dropna(subset=[label])
            new_tiles_with_labels = len(self.labels)
            if original_tiles_with_labels != new_tiles_with_labels:
                print(f"====== Removed {new_tiles_with_labels - original_tiles_with_labels} tiles as they did not have a label for {self.label}")

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
            
        if self.tensor_per_wsi:
            self.labels['case_dot'] = self.labels['case'] + '/' + self.labels['dot_id']
            self.labels = self.labels.groupby('case_dot').mean().reset_index() # We end up with a single row per patient, and the mean of all labels, which is the label
            print(f"Finally, we use a tensor per patient, meaning we now have {len(self.labels.index)} files.")

        self.transform = transform

    def create_graph_from_grid(self, grid: torch.tensor, graph_type: str) -> torch.tensor:
        """Takes a grid of features and transforms it into a graph

        Args:
            grid (torch.tensor): Grid loaded from disk, holds the feature vectors in a spatial grid related to the WSI
            graph_type (str): Argument specifying the type of graph to be returned. For now, only ['adj_matrix']. If we subsample 500 tiles, this will have 25,000 elements

        Returns:
            torch.tensor: Graph of specified type
        """

    def __len__(self):
        full_data_len = len(self.labels.index)
        return full_data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.labels.iloc[idx]

        
        if self.label:
            label=row[self.label]
        else:
            label=[]

        # case_id = str(self.labels.at[idx, "case"])
        # dot_id = self.labels.at[idx, "dot_id"]
        # tile_num = self.labels.at[idx, "num"]
        # patient_id = self.dot_id_to_tcga_id[dot_id.split('-')][2]

        # label= self.labels.at[idx, "msi"]

        if not self.tensor_per_wsi:
            case_id = str(row['case'])
            dot_id = row['dot_id']

            tile_num = row['num']
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
            case_dot = row['case_dot']
            case_id, dot_id = case_dot.split('/')
            
            if 'subsample' in self.csv_file:
                path_to_data = self.csv_file.split('_')
                n = path_to_data[path_to_data.index('subsample')+1].rstrip('.csv')
                append_aggregate_with = f"_subsample_{n}"
            else:
                append_aggregate_with = ""
            if not self.load_tensor_grid:
                img_name = os.path.join(self.root_dir, f'case-{case_id}',
                                        f"pid_{dot_id}_tile_vectors_extractor_{self.precomputed_from_run}{append_aggregate_with}.pt")
            else:
                img_name = os.path.join(self.root_dir, f'case-{case_id}',
                                        f"pid_{dot_id}_tile_grid_extractor_{self.precomputed_from_run}{append_aggregate_with}.pt")
            patient_id = case_id
        if self.precomputed or self.tensor_per_wsi:
            tile = torch.load(img_name, map_location='cpu')
            if self.load_tensor_grid:
            # the tile was initially saved as WxHxC, yet PyTorch wants CxWxH
            # also, we add contiguous to make sure the bits are close to each other in memory
                if not self.stack_grid:
                    target_size=244
                    #tile = tile.permute(2,0,1).contiguous()
                    tile = tile.permute(2,0,1)
                    w, h = tile.shape[-2:]
                    if w < target_size:
                        pad_w = target_size-w # int() floors, yet we want to get at least the target size
                    else:
                        pad_w = 0
                    if h < target_size:
                        pad_h = target_size-h
                    else:
                        pad_h = 0
                    # Note that the padding argument in F.pad() pads up on either side, and the first arguments pads the last dimension. so
                    # (1,1) pads 1 on top and 1 on bottom of h
                    # (1,1,2,2) pads 1 top, 1 bottom on h,  2 left, 2 right on w
                    # Here, we pad only on ONE side, to avoid any difficulties with an uneven difference between target and current
                    # Anyway, with any of the networks we use, the location will not matter. (CNN / GCNN)
                    tile = torch.nn.functional.pad(tile, (pad_h, 0, pad_w, 0), 'constant', 0) # zero-padding up to target size to make it survive the convolutions
                else:
                    target_size = 550 # since we subsample 500 tiles, this will pad up every image with zero-tensors, 
                                      # but often not by too much (this helps the network to learn that a zero-tensor is never meaningful information)

                    # Index the tile by feature vectors that do not have std=0 AND sum=0 (meaning it's a zero-tensor). 
                    # This already flattens it, as otherwise the object wouldn't make sense according to torch
                    # Permute to make it Cx(WxH)
                    tile = tile[((tile.float().std(dim=2) != 0) | (tile.sum(dim=2) != 0))]
                    tile = tile.permute(1,0)
                    stack_size = tile.shape[1]
                    if stack_size < target_size:
                        # always the case..
                        pad = target_size - stack_size
                        tile = torch.nn.functional.pad(tile, (pad, 0), 'constant', 0) # the tensor is Cx(HxW), we want to pad so that # pixels is same for all, so that's the last channel in shape, so we give a tuple for that   
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
