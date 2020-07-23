import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from experiment import ex
from model import load_model, save_model
from utils import post_config_hook

from modules import LogisticRegression
from modules.deepmil import Attention
from modules.transformations import TransformsSimCLR
from modules.losses.focal_loss import FocalLoss
from modules.Splitter import split_indices_by_patient, split_indices, get_train_val_indices

from msidata.dataset_msi import PreProcessedMSIDataset as dataset_msi
from msidata.save_feature_vectors import infer_and_save
from msidata.dataset_msi_features_with_patients import PreProcessedMSIFeatureDataset
from msidata.dataset_tcga_tiles import TiledTCGADataset as dataset_tcga

import pandas as pd
import time
import datetime
import os
import json

from sklearn import metrics


def infer(args, loader, context_model, device):
    # get encoding of images
    feature_vector, labels_vector, patients, imgs = [], [], [], []
    for step, (x, y, patient, img_name) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            if args.logistic_extractor == 'simclr':
                h, z = context_model(x)
            else:
                h = context_model(x)
        h = h.detach()
        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

        patients+=patient
        imgs+=img_name

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector, patients, imgs


def get_features(args, context_model, train_loader, val_loader, test_loader, device):
    train_X, train_y, train_patients, train_imgs = infer(args, train_loader, context_model, device)
    val_X, val_y, val_patients, val_imgs = infer(args, val_loader, context_model, device)
    test_X, test_y, test_patients, test_imgs = infer(args, test_loader, context_model, device)
    return train_X, train_y, val_X, val_y, test_X, test_y, train_patients, train_imgs, val_patients, val_imgs, test_patients, test_imgs


def create_data_loaders_from_arrays(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, train_patients, train_imgs, val_patients, val_imgs, test_patients, test_imgs):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train), torch.Tensor(train_patients)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    val = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val), torch.Tensor(val_patients)
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test), torch.Tensor(test_patients)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader


def train(args, train_loader, val_loader, extractor, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, data in enumerate(train_loader):
        optimizer.zero_grad()
        x =  data[0].to(args.device) 
        y = data[1].to(args.device)

        if not (args.precompute_features or args.use_precomputed_features):
            if args.freeze_encoder:
                extractor.eval()
                with torch.no_grad():
                    out = extractor.forward(x)
            else:
                extractor.train()
                out = extractor.forward(x)
        
            if args.logistic_extractor=='simclr': # Simclr returns (h, z)
                h = out[0]
                x = h
            else:  # Torchvision models return h
                h = out
                x = h  # We name it x, since it's the input for the logistic regressors, and it saves some memory
        
        model.train()
        if args.classification_head == 'logistic':
            output = model(x)
            if not args.use_focal_loss:
                loss = criterion(output, y)
            else: # use focal loss
                focal = FocalLoss(args.focal_loss_alpha, args.focal_loss_gamma)
                loss = torch.nn.functional.cross_entropy(output,y, reduction='none')
                loss = focal(loss)

            predicted = output.argmax(1)
            acc = (predicted == y).sum().item() / y.size(0)
            loss_epoch += loss.item()

        elif args.classification_head == 'deepmil':
            Y_prob, Y_hat, A = model.forward(x)
            loss = criterion(Y_prob, y)
            train_loss = loss.item()
            loss_epoch += train_loss
            error, _ = model.calculate_classification_error(y, Y_hat)
            acc = 1. - error

        if not args.freeze_encoder:
            previous_weights_of_last_layer = list(extractor.state_dict().values())[-1]
        
        accuracy_epoch += acc
        loss.backward()         # Depending on the setup: Computes gradients for classifier and possibly extractor
        optimizer.step()        # Can update both the classifier and the extractor, depending on the options
        if step % 20 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Training...")

  
        if not args.freeze_encoder:
            new_weights_of_last_layer = list(extractor.state_dict().values())[-1]

            assert(not torch.eq(new_weights_of_last_layer, previous_weights_of_last_layer).all()), "The weights are not being updated!"

    return loss_epoch, accuracy_epoch


def validate(args, loader, extractor, model, criterion, optimizer):
    loss_epoch, accuracy_epoch = 0, 0
    labels, preds, patients, img_names = [], [], [], []

    model.eval()

    for step, data in enumerate(loader):
        model.zero_grad()

        x = data[0]
        y = data[1]
        patient = data[2]

        x = x.to(args.device)
        y = y.to(args.device)

        if not (args.precompute_features or args.use_precomputed_features):
            # x is an image not yet a feature vector
            extractor.eval()

            with torch.no_grad():
                out = extractor.forward(x)

            if args.logistic_extractor=='simclr': # Simclr returns (h, z)
                h = out[0]
                x = h
            else: # Torchvision models return h
                h = out
                x = h  # We name it x, since it's the input for the logistic regressors
               
        else:
            # x is already a feature vector, and can go straight into the classifier
            pass

        if args.classification_head == 'logistic':
            with torch.no_grad():
                output = model(x)
                if not args.use_focal_loss:
                    loss = criterion(output, y)
                else:
                    # use focal loss
                    focal = FocalLoss(args.focal_loss_alpha, args.focal_loss_gamma)
                    loss = torch.nn.functional.cross_entropy(output,y, reduction='none')
                    loss = focal(loss)

                predicted = output.argmax(1)
                acc = (predicted == y).sum().item() / y.size(0)
                loss_epoch += loss.item()
                preds += predicted.cpu().tolist()

        elif args.classification_head == 'deepmil':
            with torch.no_grad():
                Y_prob, Y_hat, A = model.forward(x)
                loss = criterion(Y_prob, y)
                train_loss = loss.item()
                loss_epoch += train_loss
                error, _ = model.calculate_classification_error(y, Y_hat)
                acc = 1. - error        
                binary_Y_prob = Y_prob.softmax(dim=1)[0][1]
                preds.append(binary_Y_prob.item())       

        accuracy_epoch += acc
        labels += y.cpu().tolist()
        
        if isinstance(patient, torch.Tensor):
            # Happens when using dataset_msi, as we return a hash, which is an integer, which is made into a tensor
            patients += patient.cpu().tolist()
        else:
            # Happens when using dataset_Msi_features_with_patients, as we return the patient id as a string, which can't be tensorfied
            patients += list(patient)

    return loss_epoch, accuracy_epoch, labels, preds, patients


def get_precomputed_dataloader(args, run_id):
    print(f"### Loading precomputed feature vectors from run id:  {run_id} ####")

    assert(args.load_patient_level_tensors and args.logistic_batch_size==1) or not args.load_patient_level_tensors, "We can only use batch size=1 for patient-level tensors, due to different size of tensors"

    if args.load_patient_level_tensors:
        sampling_strategy='patient'
    else:
        sampling_strategy='tile'

    if args.dataset=='msi-kather':

        train_dataset = PreProcessedMSIFeatureDataset(
            root_dir=args.path_to_msi_data, 
            transform=None, 
            data_fraction=1,
            sampling_strategy=sampling_strategy,
            append_img_path_with=f'_{run_id}',
            tensor_per_patient=args.load_patient_level_tensors,
            seed=args.seed
            
        )
        test_dataset = PreProcessedMSIFeatureDataset(
            root_dir=args.path_to_test_msi_data, 
            transform=None, 
            data_fraction=1,
            sampling_strategy=sampling_strategy,
            append_img_path_with=f'_{run_id}',
            tensor_per_patient=args.load_patient_level_tensors,
            seed=args.seed
        )

    elif args.dataset == 'msi-tcga':
        train_dataset = dataset_tcga(
            csv_file=args.path_to_msi_data, 
            root_dir=args.root_dir_for_tcga_tiles, 
            transform=None,
            precomputed=True,
            precomputed_from_run=run_id,
            sampling_strategy=sampling_strategy,
            tensor_per_patient=args.load_patient_level_tensors
            )     
        test_dataset = dataset_tcga(
            csv_file=args.path_to_msi_data, 
            root_dir=args.root_dir_for_tcga_tiles, 
            transform=None,
            precomputed=True,
            precomputed_from_run=run_id,
            sampling_strategy=sampling_strategy,
            tensor_per_patient=args.load_patient_level_tensors
            )

    train_indices, val_indices = get_train_val_indices(train_dataset, val_split=args.validation_split)
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=val_sampler
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader, test_loader

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def write_hparams(writer, config, metric):
    config_vars = vars(config)
    for key in config_vars.keys():
        if isinstance(config_vars[key], bool):
            if config_vars[key]:
                config_vars[key] = 1
            else:
                config_vars[key] = 0
    
    del config_vars['device']

    writer.add_hparams(config_vars, metric)

@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    if 'train_extractor_on_generated_labels' in vars(args).keys():
        if args.train_extractor_on_generated_labels:
            assert('generated_labels_id' in vars(args).keys()), "Please set the ID of the run that generated the labels"
            assert('generated_label' in vars(args).keys()), "Please set the label you want to use"
            assert(args.generated_label != 'label'), "Please set a non-standard label, otherwise we're not doing anything interesting"
            assert(not args.freeze_encoder), "If we want to finetune, we should not freeze the encoder"
            assert(not args.precompute_features), "If we want to finetune, we should not precompute any features. We should run the images through the encoder"
            assert(args.classification_head == 'logistic'), "We want to do tilewise predictions with our new tile-level labels!"
            with open(f'./logs/{args.generated_labels_id}/config.json') as f:
                config_of_label_creation = json.load(f)
            assert(args.seed == config_of_label_creation['seed']), f"Current seed: {args.seed}. Seed during label creation of run {args.generated_labels_id}: {config_of_label_creation['seed']}. Ensure they are equal to get the same train-test split"



    tb_dir = os.path.join(args.out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    if 'train_extractor_on_generated_labels' in vars(args).keys():
        if args.train_extractor_on_generated_labels:
            label = args.generated_label
            load_labels_from_run = args.generated_labels_id
        else:
            label = 'label'
            load_labels_from_run=''
    else:
        label = 'label'
        load_labels_from_run=''

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
        args.data_pretrain_fraction=1    
        assert ('.csv' in args.path_to_msi_data), "Please provide the tcga .csv file in path_to_msi_data"
        assert ('root_dir_for_tcga_tiles' in vars(args).keys()), "Please provide the root dir for the tcga tiles"
        train_dataset = dataset_tcga(
            csv_file=args.path_to_msi_data, 
            root_dir=args.root_dir_for_tcga_tiles, 
            transform=TransformsSimCLR(size=224).test_transform
            )     
        test_dataset = dataset_tcga(
            csv_file=args.path_to_test_msi_data, 
            root_dir=args.root_dir_for_tcga_tiles, 
            transform=TransformsSimCLR(size=224).test_transform
            )
    else:
        raise NotImplementedError

    # Get the train and validation samplers
    train_indices, val_indices = get_train_val_indices(train_dataset, val_split=args.validation_split)
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)  


    # Get the extractor
    if args.logistic_extractor == 'byol':
        # We get the rn18 backbone with the loaded state dict
        print("Loading BYOL model ... ")
        _, _, _, extractor, n_features = load_model(args, None, reload_model=args.reload_model, model_type=args.logistic_extractor)
    else:
        extractor, _, _ = load_model(args, None, reload_model=args.reload_model, model_type=args.logistic_extractor)

    extractor = extractor.to(args.device)

    # Freeze encoder if asked for
    if args.freeze_encoder:
        extractor.eval()
    else:
        extractor.train()

    ## Logistic Regression
    # n_classes = 10  # stl-10
    n_classes = 2  # MSI VS MSS
    
    # Get number of features that feature extractor produces
    if args.logistic_extractor == 'simclr':
        n_features = extractor.n_features
    elif args.logistic_extractor == 'byol':
        # We returned n_features in load_model()..
        pass
    else:
        n_features = {'imagenet-resnet18': 512, 'imagenet-resnet50': 2048, 'imagenet-simclr_v1_x1_0': 1024}[args.logistic_extractor]

    ## Get Classifier
    if args.classification_head == 'logistic':
        model = LogisticRegression(n_features, n_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    elif args.classification_head == 'deepmil':
        model = Attention(hidden_dim=n_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.deepmil_lr, betas=(0.9, 0.999), weight_decay=args.deepmil_reg)

        
    model = model.to(args.device)

    print(model)
    
    criterion = torch.nn.CrossEntropyLoss()

    ### ============= GET THE CORRECT DATA LOADERS =============  ###

    if args.precompute_features or not args.use_precomputed_features:
        # If we precompute features, we need an image loader
        # If we use precomputed features, we don't need an image loader
        # If we don't use any preocomputed features, which happens if we finetune, we do want an image loader 

        assert not (args.precompute_features and args.use_precomputed_features), "Ambiguous config. Precompute features or use precomputed features?"

        drop_last = not (args.precompute_features and not args.precompute_features_in_memory) # if we precompute features, but NOT in memory, do not drop last

        #TODO ISNT THIS EXACTLY THE SAME AS THE LOADERS ABOVE? CAN SKIP THIS?

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.logistic_batch_size,
            drop_last=drop_last,
            num_workers=args.workers,
            sampler=train_sampler
        )
        val_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.logistic_batch_size,
            drop_last=drop_last,
            num_workers=args.workers,
            sampler=val_sampler
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.logistic_batch_size,
            drop_last=drop_last,
            num_workers=args.workers
        )
     

    if args.precompute_features:
        # We need the image loader defined above 
   
        if args.precompute_features_in_memory:
            print("### Creating features from pre-trained context model ###")
            
            (train_X, train_y, val_X, val_y, test_X, test_y, train_patients, train_imgs, val_patients, val_imgs, test_patients, test_imgs) = get_features(
                args, extractor, train_loader, val_loader, test_loader, args.device
            )

            arr_train_loader, arr_val_loader, arr_test_loader = create_data_loaders_from_arrays(
                train_X, train_y, val_X, val_y, test_X, test_y, args.logistic_batch_size, train_patients, train_imgs, val_patients, val_imgs, test_patients, test_imgs
            )
        else:
            print("### Creating and saving features from pre-trained context model ###")
            print(args.data_testing_train_fraction)
            assert args.data_testing_train_fraction == 1 and args.data_testing_test_fraction == 1, "Bugs might occur when we do not save feature vectors for all data due to sampling issues"

            run_id = args.out_dir.split('/')[-1] # "./logs/pretrain/<id>"

            infer_and_save(loader=train_loader, context_model=extractor, device=args.device, append_with=f'_{run_id}', model_type=args.logistic_extractor)
            infer_and_save(loader=val_loader, context_model=extractor, device=args.device, append_with=f'_{run_id}', model_type=args.logistic_extractor)
            infer_and_save(loader=test_loader, context_model=extractor, device=args.device, append_with=f'_{run_id}', model_type=args.logistic_extractor)
            
            # Overwriting previous variable names to reduce memory load
            train_loader, val_loader, test_loader = get_precomputed_dataloader(args, run_id)

            arr_train_loader, arr_val_loader, arr_test_loader = train_loader, val_loader, test_loader

    elif args.use_precomputed_features:
        # Did not need any image loader, as we create a new vector-based dataset
        
        assert (args.use_precomputed_features_id), 'Please set the run ID of the features you want to use'
        print(f"Removing SIMCLR model from memory, as we use precomputed features..")
        del extractor
        extractor = None
        arr_train_loader, arr_val_loader, arr_test_loader = get_precomputed_dataloader(args, args.use_precomputed_features_id)
    else:
        # We use the image loader as defined above
        arr_train_loader, arr_val_loader, arr_test_loader = train_loader, val_loader, test_loader


    ### ============= TRAINING =============  ###
    val_losses = []
    val_roc = []
    for epoch in range(args.logistic_epochs):

        # Train for 1 epoch
        loss_epoch, accuracy_epoch = train(args, arr_train_loader, arr_val_loader, extractor, model, criterion, optimizer)

        writer.add_scalar("loss/train", loss_epoch / len(arr_train_loader), epoch)

        # Evaluate on validation set
        if (epoch+1) % args.evaluate_every == 0:
            # evaluate
            val_loss, val_accuracy, val_labels, val_preds, val_patients = validate(args, arr_val_loader, extractor, model, criterion, optimizer)
            val_losses.append(val_loss)

            # COMPUTE ROCAUC PER PATIENT. If we use a patient-level prediction, group and mean doesn't do anything. 
            # If we use a tile-level prediciton, group and mean create a fraction prediction
            val_data = pd.DataFrame(data={'patient': val_patients, 'labels': val_labels, 'preds': val_preds})
            dfgroup = val_data.groupby(['patient']).mean()
            labels = dfgroup['labels'].values
            preds = dfgroup['preds'].values
            rocauc=metrics.roc_auc_score(y_true=labels, y_score=preds)

            writer.add_scalar("loss/val", val_loss / len(arr_val_loader), epoch)
            writer.add_scalar("rocauc/val", rocauc, epoch)

            val_roc.append(rocauc)

            args.current_epoch = epoch+1

            # Save model with each evaluation
            if extractor:
                # If we do not have an extractor, e.g. when we use precomputed features
                save_model(args, extractor, None, prepend='extractor_')
            # Save classification model, which is the primary model being trained here
            save_model(args, model, None, prepend='classifier_')
        
            print(
            f"{datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S')} | \
                Epoch [{epoch+1}/{args.logistic_epochs}]\t \
                    Val Loss: {val_loss / len(arr_val_loader)}\t \
                        Accuracy: {val_accuracy / len(arr_val_loader)}\t \
                            ROC AUC: {rocauc}"
            )

    ### ============= TESTING =============  ###

    best_model_num = np.argmax(val_roc)
    best_model_epoch = best_model_num * args.evaluate_every + args.evaluate_every

    print(f'Validation ROCs: {val_roc}\nValidation loss: {val_losses}.\nBest performance by model @ epoch # {best_model_epoch}')

    if extractor:
        extractor_fp = os.path.join(args.out_dir, "extractor_checkpoint_{}.tar".format(best_model_epoch))
        extractor.load_state_dict(torch.load(extractor_fp, map_location=args.device.type))
    model_fp = os.path.join(args.out_dir, "classifier_checkpoint_{}.tar".format(best_model_epoch))
    model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        
    loss_epoch, accuracy_epoch, labels, preds, patients = validate(
        args, arr_test_loader, extractor, model, criterion, optimizer
    )

    final_data = pd.DataFrame(data={'patient': patients, 'labels': labels, 'preds': preds})

    humane_readable_time = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d-%H-%M-%S')

    print(f'This is the out dir {args.out_dir}')
    final_data.to_csv(f'{args.out_dir}/regression_output_epoch_{epoch+1}_{humane_readable_time}.csv')

    dfgroup = final_data.groupby(['patient']).mean()
    labels = dfgroup['labels'].values
    preds = dfgroup['preds'].values
    rocauc=metrics.roc_auc_score(y_true=labels, y_score=preds)

    writer.add_scalar("loss/test", loss_epoch / len(arr_test_loader))
    writer.add_scalar("rocauc/test", rocauc)

    write_hparams(writer, args, {'hparam/rocauc': rocauc })

    _run.log_scalar('test/rocauc', rocauc, 0)

    print(
        f"======\n[Final test with best model]\t ROCAUC: {rocauc} \t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}"
    )

