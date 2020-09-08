import torch
from viz.visualizer import Visualizer

from modules.deepmil import Attention
from msidata.dataset_msi_features_with_patients import PreProcessedMSIFeatureDataset
from testing.logistic_regression import get_precomputed_dataloader

import argparse

from experiment import ex
from utils import post_config_hook





@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get test data to be visualized
    _, test_loader = get_precomputed_dataloader(args, args.use_precomputed_features_id)

    # Load model to be used
    model = Attention()

    # Initialize visualizer.. not necessary?
     
    viz = Visualizer()

    viz.visualize_first_patient(test_loader, model, method='deepmil')
    print('done')

    # for step, data in enumerate(loader):
    #     optimizer.zero_grad()
    #     x = data[0]
    #     y = data[1]





if __name__=="__main__":
    main()