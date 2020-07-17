from experiment import ex
from utils import post_config_hook
import torch
import argparse

from testing.logistic_regression import get_precomputed_dataloader
from model import load_model, save_model

import pandas as pd
import os


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert (args.use_precomputed_features_id), 'Please set the run ID of the features you want to use'
    assert (args.load_patient_level_tensors), 'Please load patient level tensors for deepmil'
    assert (args.classification_head == 'deepmil'), 'Currently this is only implemented for deepmil'
    assert (args.dataset == 'msi-kather'), 'Currently this is only implemented for kather-msi data'

    # Get a patient-level tensor loader
    train_loader, val_loader, test_loader = get_precomputed_dataloader(args, args.use_precomputed_features_id)

    # Also, get the original data file
    original_labels = pd.read_csv(os.path.join(args.path_to_msi_data, 'data.csv'))
    original_labels['patient_id'] = original_labels.apply(lambda x: x['img'].split('/')[1].split('-')[4], axis=1)

    # Get a pretrained deepmil classifier
    classifier, _, _ = load_model(args, None, reload_model=False, model_type=args.classification_head)
    classifier = classifier.to(args.device)

    for step, data in enumerate(train_loader):
          
        # We get an unstructured compressed image for a single patient. i.e. a 1x{# tiles}x{h=512} vector
        x, y, patient = data[0].to(args.device), data[1].to(args.device), data[2][0]
        x.requires_grad = True

        # ----------- Predict for train
        # Run through classifier
        Y_out, Y_hat, A = classifier.forward(x)
        binary_Y_prob = Y_out.softmax(dim=1)[0][1]
        
        # ----------- Compute the labels
        # Get MSI gradient labels
        frac = Y_out[0][1] / (Y_out[0].sum())
        frac.backward()
        dMSIdA = classifier.A_grad    

        # Save the predictions into the original data file
        # The saved feature vectors have the same name as the images
        # The aggregated feature vector takes the same dataloader, groups by patient, and stacks them
        # So this SHOULD be the right order...

        original_labels.loc[original_labels['patient_id']==patient, 'attention'] = A.flatten().detach().cpu().numpy()
        original_labels.loc[original_labels['patient_id']==patient, 'dMSIdA'] = dMSIdA.flatten().detach().cpu().numpy()
        original_labels.loc[original_labels['patient_id']==patient, 'a_dMSIdA'] = (A.flatten()*dMSIdA.flatten()).detach().cpu().numpy()

    # Save the labels to data.csv
    # save it in the same dir as data.csv, but as data_{run_id}.csv
    run_id = args.out_dir.split('/')[-1] # "./logs/pretrain/<id>"
    original_labels = original_labels.dropna() # Drop the prediction on the validation set
    original_labels.to_csv(os.path.join(args.path_to_msi_data, f'data_{run_id}.csv'))

  

