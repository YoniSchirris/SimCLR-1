import os
os.environ["OMP_NUM_THREADS"] = "1"
import random
import torch
torch.set_num_threads(1)
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
from msidata.save_feature_vectors import infer_and_save, aggregate_patient_vectors
from msidata.dataset_msi_features_with_patients import PreProcessedMSIFeatureDataset
from msidata.dataset_tcga_tiles import TiledTCGADataset as dataset_tcga

import matplotlib.pyplot as plt

import pandas as pd
import time
import datetime
import json

from sklearn import metrics


def infer(args, loader, context_model, device):
    # get encoding of images
    feature_vector, labels_vector, patients, imgs = [], [], [], []
    for step, (x, y, patient, img_name, _) in enumerate(loader):
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


def train(args, train_loader, val_loader, extractor, model, criterion, optimizer, val_losses, val_roc, global_step, epoch, writer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, data in enumerate(train_loader):
        global_step += 1
        print(f"[ Step {global_step} / {len(train_loader)} ] @ {datetime.datetime.now()}")
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
            y = y.long()
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

        elif args.classification_head == 'linear':
            output = model(x).flatten() # output \in R^(batch_size*1)
            loss = criterion(output.flatten(), y.flatten())
            loss_epoch += loss.item() 
            acc = 0 # accuracy is not meaningful here
            
        elif args.classification_head == 'deepmil':
            y = y.long() # Might be a float if we use TCGA since we do a groupby.mean() operation
            Y_prob, Y_hat, A = model.forward(x)
            loss = criterion(Y_prob, y)
            train_loss = loss.item()
            loss_epoch += train_loss
            error, _ = model.calculate_classification_error(y, Y_hat)
            acc = 1. - error

        elif args.classification_head == 'linear-deepmil':
            Y_prob, Y_hat, A = model.forward(x) # Y_prob == Y_hat, in the linear case
            y = y.flatten() # shape([1, 1]) -> shape([1])
            loss = criterion(Y_prob.flatten(), y.flatten())
            train_loss = loss.item()
            loss_epoch += train_loss
            acc = 0 # meaningless here

        elif 'cnn' in args.classification_head:
            y = y.long()
            out = model.forward(x)
            loss = criterion(out, y)
            predicted = out.argmax(1)
            acc = (predicted == y).sum().item() / y.size(0)
            loss_epoch += loss.item()

        if not args.freeze_encoder and args.debug:
            previous_weights_of_last_layer = [param.clone().detach() for param in extractor.parameters()][-2]

        accuracy_epoch += acc
        loss.backward()         # Depending on the setup: Computes gradients for classifier and possibly extractor
        optimizer.step()        # Can update both the classifier and the extractor, depending on the options
       
        if not args.freeze_encoder and args.debug:
            new_weights_of_last_layer = [param.clone().detach() for param in extractor.parameters()][-2]
            assert(not torch.eq(new_weights_of_last_layer, previous_weights_of_last_layer).all()), "The weights are not being updated!"

         # Evaluate on validation set
        if global_step % args.evaluate_every == 0:
            # evaluate
            val_loss, val_accuracy, val_labels, val_preds, val_patients, _, _, _, _, _ = validate(args, val_loader, extractor, model, criterion, optimizer, final_test=False)
            val_losses.append(val_loss / len(val_loader))

            if args.classification_head not in ['linear', 'linear-deepmil']:
                # COMPUTE ROCAUC PER PATIENT. If we use a patient-level prediction, group and mean doesn't do anything. 
                # If we use a tile-level prediciton, group and mean create a fraction prediction
                val_data, rocauc = compute_roc_auc(args, val_patients, val_labels, val_preds, save_curve=False)
                writer.add_scalar("loss/val", val_loss / len(val_loader), epoch)
                writer.add_scalar("loss/train", loss.cpu().item(), epoch)
                writer.add_scalar("rocauc/val", rocauc, epoch)
                val_roc.append(rocauc)
                print(f"{datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S')} | Epoch [{epoch+1}/{args.logistic_epochs}]\tStep [{step}/{len(train_loader)}]\t Train loss: {loss.cpu().item()}\tVal Loss: {val_loss / len(val_loader)}\tAccuracy: {val_accuracy / len(val_loader)}\tROC AUC: {rocauc}")
            else:
                # COMPUTE R2 PER PATIENT.
                val_data, r2 = compute_r2(val_patients, val_labels, val_preds)
                writer.add_scalar("loss/val", val_loss / len(val_loader), epoch)
                writer.add_scalar("r2/val", r2, epoch)
                val_roc.append(r2)
                print(f"{datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S')} | Epoch [{epoch+1}/{args.logistic_epochs}]\tStep [{step}/{len(train_loader)}]\tVal Loss: {val_loss / len(val_loader)}\tAccuracy: {val_accuracy / len(val_loader)}\tR2: {r2}")
                
            # Save model with each evaluation
            args.global_step = global_step
            if extractor:
                # We don't always have an extractor, e.g. when we use precomputed features
                save_model(args, extractor, None, prepend='extractor_')
            # Save classification model, which is the primary model being trained here
            save_model(args, model, None, prepend='classifier_')
        

    return loss_epoch, accuracy_epoch, val_losses, val_roc, global_step


def validate(args, loader, extractor, model, criterion, optimizer, final_test=False):
    loss_epoch, accuracy_epoch = 0, 0
    labels, preds, patients, img_names, attentions, attention_gradients, subsample_indices, predicted_probs = [], [], [], [], [], [], [], []

    model.eval()

    for step, data in enumerate(loader):
        model.zero_grad()

        x = data[0]
        y = data[1]
        patient = data[2]
        img_names += data[3] # a list of strings, each of one patient -> e.g. ['grid_123.pt', 'grid_234.pt', 'grid_345.pt', 'grid_456.pt']
        subsample_indices += data[4] # a list of indices per patient -> e.g. [ [1, 2, 6, 7] , [4, 6, 8, 10] , [1, 14, 17, 200] ]

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
            y = y.long()
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
                predicted_prob = torch.nn.functional.softmax(output, dim=1)
                acc = (predicted == y).sum().item() / y.size(0)
                loss_epoch += loss.item()
                preds += predicted.cpu().tolist()
                predicted_probs += predicted_prob.cpu().tolist()

        elif args.classification_head == 'linear':
            with torch.no_grad():
                output = model(x).flatten()
                loss = criterion(output.flatten(), y.flatten())
                predicted = output
                acc = 0 # meaningless for linear regression
                loss_epoch += loss.item()
                preds += predicted.cpu().tolist()

        elif args.classification_head == 'deepmil':
            if final_test:
                x.requires_grad = True
            with torch.set_grad_enabled(final_test): ## We do want gradients for the final test, in order to do some nice visualization with the gradients & attention

                y = y.long() # Might be a float when using TCGA data due to groupby.mean() operator
                Y_prob, Y_hat, A = model.forward(x)
                loss = criterion(Y_prob, y)
                train_loss = loss.item()
                loss_epoch += train_loss
                error, _ = model.calculate_classification_error(y, Y_hat)
                acc = 1. - error        
                binary_Y_prob = Y_prob.softmax(dim=1)[:,1] # Get the probability of it being class 1
                preds += binary_Y_prob.cpu().tolist() 
                if final_test:
                    # print(f"Shape of x: {x.shape}")
                    # print(f"Shape of A: {A.shape}")
                    # print(f"Shape of binary_Y_prob: {binary_Y_prob}")

                    ## Get the DeepMIL attention gradients
                    for i, patient_probability in enumerate(binary_Y_prob): # Pytorch can't do elemnt-wise backwards

                        # print(f"Shape of patient probability: {patient_probability}")
                        
                        patient_probability.backward(retain_graph=True) # So we do a backward pass per patient
                        dPositiveClassdA = model.A_grad[i].flatten() # Which means we now get a gradient from attention to class probability for each tile for the patient
                        # since attention is batch x tiles x 1, we want the index of the patient we look at now, and flatten it.
                        
                        # print(f"Shape of a single dPositiveClassdA: {dPositiveClassdA.shape}")
                        # print(f"{i}: {dPositiveClassdA}")
                        attention_gradients.append(dPositiveClassdA.cpu().tolist()) # We save these as a list per patient
                        model.zero_grad() # We remove the created gradients, so that they are not accumulated

                    attentions += A.flatten(start_dim=1, end_dim=-1).cpu().tolist() # save attention as a list of attentions per patient

        elif args.classification_head == 'linear-deepmil':
            with torch.no_grad():
                Y_prob, Y_hat, A = model.forward(x)
                y = y.flatten() # torch.size([1,1]) -> torch.size([1])
                loss = criterion(Y_prob.flatten(), y.flatten())
                train_loss = loss.item()
                loss_epoch += train_loss
                acc = 0
                preds += Y_prob.cpu().tolist()

        elif 'cnn' in args.classification_head:
            with torch.no_grad():
                y = y.long()
                out = model.forward(x)
                loss = criterion(out, y)
                predicted = out.argmax(1)
                acc = (predicted == y).sum().item() / y.size(0)
                loss_epoch += loss.item()
                preds += predicted.cpu().tolist()

        accuracy_epoch += acc
        labels += y.cpu().tolist()
        
        if isinstance(patient, torch.Tensor):
            # Happens when using dataset_msi, as we return a hash, which is an integer, which is made into a tensor
            patients += patient.cpu().tolist()
        else:
            # Happens when using dataset_Msi_features_with_patients, as we return the patient id as a string, which can't be tensorfied
            patients += list(patient)

    return loss_epoch, accuracy_epoch, labels, preds, patients, attentions, attention_gradients, img_names, subsample_indices, predicted_probs


def get_precomputed_dataloader(args, run_id):
    print(f"### Loading precomputed feature vectors from run id:  {run_id} ####")

    if args.dataset in ['msi-tcga', 'basis']:
        assert ('load_wsi_level_tensors' in vars(args).keys()), "For TCGA we switched to WSI-level tensors. Please add 'load_wsi_level_tensors' (bool) to your config file"

    #assert(args.load_patient_level_tensors and args.logistic_batch_size==1) or not args.load_patient_level_tensors, "We can only use batch size=1 for patient-level tensors, due to different size of tensors"

    if args.load_patient_level_tensors:
        sampling_strategy='patient'
    else:
        sampling_strategy='tile'

    if 'deepmil' in args.classification_head:
        stack_grid = True
    else:
        stack_grid = False

    if args.dataset=='msi-kather':

        train_dataset = PreProcessedMSIFeatureDataset(
            root_dir=args.path_to_msi_data, 
            transform=None, 
            data_fraction=1,
            sampling_strategy=sampling_strategy,
            append_img_path_with=f'_{run_id}',
            tensor_per_patient=args.load_patient_level_tensors,
            seed=args.seed,
            pad_tiles=args.logistic_batch_size > 1
            
        )
        test_dataset = PreProcessedMSIFeatureDataset(
            root_dir=args.path_to_test_msi_data, 
            transform=None, 
            data_fraction=1,
            sampling_strategy=sampling_strategy,
            append_img_path_with=f'_{run_id}',
            tensor_per_patient=args.load_patient_level_tensors,
            seed=args.seed,
            pad_tiles=args.logistic_batch_size > 1
        )

    elif args.dataset in ['msi-tcga', 'basis']:
        train_dataset, test_dataset, val_dataset = [dataset_tcga(
            args=args,
            csv_file=args.path_to_msi_data, 
            root_dir=args.root_dir_for_tcga_tiles, 
            transform=None,
            precomputed=True,
            precomputed_from_run=run_id,
            tensor_per_wsi=args.load_wsi_level_tensors,
            split_num=args.kfold,
            label=args.ddr_label,
            split=current_split,
            load_tensor_grid=args.load_tensor_grid,
            stack_grid=stack_grid,
            load_normalized_tiles=args.load_normalized_tiles,
            dataset=args.dataset
            ) for current_split in ['train', 'test', 'val']]

    if args.dataset=='msi-kather':
        train_indices, val_indices = get_train_val_indices(train_dataset, val_split=args.validation_split)
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        val_dataset=train_dataset
    elif args.dataset in ['msi-tcga','basis']:
        train_sampler=None
        val_sampler=None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
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

    assert (len(train_loader) != len(test_loader))
    assert (len(train_loader) != len(val_loader))
    assert (len(val_loader) != len(test_loader))

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

def compute_roc_auc(args, patients, labels, preds, save_curve=False, epoch=0):
    data = pd.DataFrame(data={'patient': patients, 'labels': labels, 'preds': preds})
    dfgroup = data.groupby(['patient']).mean()
    labels = dfgroup['labels'].values
    preds = dfgroup['preds'].values
    print(f"Y_true: {labels}")
    print(f"Y_score: {preds}")
    rocauc = metrics.roc_auc_score(y_true=labels, y_score=preds)

    if save_curve:
        fpr, tpr, threshold = metrics.roc_curve(labels, preds)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % rocauc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        plt.savefig(f'{args.out_dir}/roc_curve_epoch_{epoch}.png')

    return data, rocauc

def compute_r2(patients, labels, preds):
    data = pd.DataFrame(data={'patient': patients, 'labels': labels, 'preds': preds})
    dfgroup = data.groupby(['patient']).mean() # Let's just try taking the mean of all the tile predictions.. for now.. if it's linear-deepmil this will keep them as is
    labels = dfgroup['labels'].values # all continues values
    preds = dfgroup['preds'].values # all continues values
    r2=metrics.r2_score(y_true=labels, y_pred=preds)
    return data, r2

def save_attention(args, img_names, attentions, attention_gradients, subsample_indices, out_dir, humane_readable_time, epoch):

    all_x_coords, all_y_coords, all_tile_names, all_patient_ids = [], [], [], []

    for img_name, subsample_indices_per_patient, attention_per_patient, attention_gradient_per_patient in zip(img_names, subsample_indices, attentions, attention_gradients):

        # read grid with tiles
        meta_img_name = img_name.replace('grid_extractor', 'grid_paths_and_coords_extractor')
        meta_img = torch.load(meta_img_name, map_location='cpu')
        if args.dataset == 'msi-tcga':
            patient_id = img_name.split('/')[5].split('case-')[1] # all img names are from the same patient ID
        elif args.dataset == 'basis':
            patient_id = img_name.split('/')[4].split('case-')[1] # this is incorrect. img_name e.g. = /project/schirris/tiled_data_large/case-PD9578a/pid_YID161_tile_grid_extractor_585.pt
        tile_names = list(meta_img[0])
        coords = meta_img[1]
        tile_names = [tile_names[int(i)] if not torch.isnan(i) else np.nan for i in subsample_indices_per_patient]
        x_coords_unsampled = coords[:,0]
        y_coords_unsampled = coords[:,1]
        x_coords = [x_coords_unsampled[int(i)] if not torch.isnan(i) else np.nan for i in subsample_indices_per_patient]
        y_coords = [y_coords_unsampled[int(i)] if not torch.isnan(i) else np.nan for i in subsample_indices_per_patient]


        # We pad the left withe [None]s, because the subsample indices might be smaller than the actual array, and therefore smaller than the attention array
        # In order to make DeepMIL work with a bigger batch_size than 1, we pad each "bag" with zero-vectors. For consistency's sake, we continued doing this when
        # subsampling. However, in this case, this would lead to a larger attention vector than there are actual images. This would not be savable.
        # However, since we want to see how much attention there is on zero-vectors (as a sanity check) we want to save this, but with empty tile names and empty
        # coordinates
        tile_names = [np.nan] * (len(attention_per_patient) - len(tile_names)) + tile_names
        x_coords = [np.nan] * (len(attention_per_patient) - len(x_coords)) + x_coords
        y_coords = [np.nan] * (len(attention_per_patient) - len(y_coords)) + y_coords
        patient_ids = [patient_id] * len(tile_names)
        

        all_x_coords += x_coords # These are all a single list of objects now
        all_y_coords += y_coords
        all_tile_names += tile_names
        all_patient_ids += patient_ids


    attention_df = pd.DataFrame({"tile_name": all_tile_names, "patient_id": all_patient_ids, "attention": np.array(attentions).flatten(), "attention_gradients": np.array(attention_gradients).flatten(), 'x': all_x_coords, 'y': all_y_coords})

    attention_df.to_csv(f'{out_dir}/attention_output_epoch_{epoch}_{humane_readable_time}.csv')


def save_probabilities(args, img_names, predicted_probs, labels, out_dir, humane_readable_time, epoch):
    data = pd.DataFrame(data={'img': img_names, 'labels': labels, 'probs': predicted_probs})
    data.to_csv(f'{out_dir}/probabilities_output_epoch_{epoch}_{humane_readable_time}.csv')


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if 'debug' not in vars(args).keys():
        args.debug = False

    if 'load_normalized_tiles' not in vars(args).keys():
        args.load_normalized_tiles = False

    if 'test_deepmil_subsample' in vars(args).keys():
        print(f"*=*=*=*=* We will perform a subsampling experiment with {args.test_deepmil_subsample} subsampled tiles *=*=*=*=*")

    if 'deepmil_intermediate_hidden' not in vars(args).keys():
        args.deepmil_intermediate_hidden = 128
    
    if 'reload_classifier' not in vars(args).keys():
        args.reload_classifier = False


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
    elif args.dataset in ["msi-tcga", "basis"]:
        args.data_pretrain_fraction=1    
        assert ('.csv' in args.path_to_msi_data), "Please provide the tcga .csv file in path_to_msi_data"
        assert ('root_dir_for_tcga_tiles' in vars(args).keys()), "Please provide the root dir for the tcga tiles"


        # TODO Add the HE normalization from config arguments to the code below
        # Then commit
        # Then push
        # Then test
        if ('he_norm' in vars(args).keys()) and ('he_norm_method' in vars(args).keys()):
            if args.he_norm:
                he_normalization=args.he_norm_method
                if he_normalization == 'macenko':
                    assert(os.path.isfile(args.he_norm_target)), f"he_norm_target: {args.he_norm_target} is not an existing file"
                    he_norm_target = args.he_norm_target
            else:
                he_normalization = ''
                he_norm_target = ''
        else:
            he_normalization = ''
            he_norm_target = ''

        tcga_transform = TransformsSimCLR(size=224, henorm=he_normalization, path_to_target_im=he_norm_target).test_transform

        train_dataset, test_dataset, val_dataset = [dataset_tcga(
            args=args,
            csv_file=args.path_to_msi_data, 
            root_dir=args.root_dir_for_tcga_tiles, 
            transform=tcga_transform,
            split_num=args.kfold,
            label=args.ddr_label,
            split=current_split,
            load_normalized_tiles=args.load_normalized_tiles,
            dataset=args.dataset
            ) for current_split in ['train', 'test', 'val']]
    else:
        raise NotImplementedError

    # Get the extractor
    if args.logistic_extractor == 'byol':
        # We get the rn18 backbone with the loaded state dict
        print("Loading BYOL model ... ")
        _, _, _, extractor, n_features = load_model(args, reload_model=args.reload_model, model_type=args.logistic_extractor)
    else:
        extractor, _, _ = load_model(args, reload_model=args.reload_model, model_type=args.logistic_extractor)

    extractor = extractor.to(args.device)

    # Set extractor to eval if asked for
    if args.freeze_encoder:
        print("===== Extractor is frozen =====")
        extractor.eval()
    else:
        print("===== Extractor is NOT frozen =====")
        extractor.train()

    # Get number of features that feature extractor produces
    if args.logistic_extractor == 'simclr':
        n_features = extractor.n_features
    elif args.logistic_extractor == 'byol':
        # We returned n_features in load_model()..
        pass
    else:
        n_features = {'imagenet-resnet18': 512, 'imagenet-resnet50': 2048, 'imagenet-shufflenetv2_x1_0': 1024}[args.logistic_extractor]

    if 'linear' in args.classification_head:
        n_classes = 1 # Doing linear regression
    else:
        n_classes = 2 # Doing logistic regression


    ## Get classifier. Logistic / linear / deepmil / linear-deepmil
    if 'reload_classifier' not in vars(args).keys():
        args.reload_classifier = False
    model, _, _ = load_model(args, reload_model=args.reload_classifier, model_type=args.classification_head, prepend='', n_features=n_features, n_classes=n_classes)

    ## Get optimizer
    if args.classification_head == 'logistic' or args.classification_head == 'linear':
        if args.freeze_encoder:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.logistic_lr)
        else:
            optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=args.logistic_lr)
    elif args.classification_head in ['deepmil', 'linear-deepmil', 'cnn-resnet18', 'cnn-densenet']:
        if args.freeze_encoder:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.deepmil_lr, betas=(0.9, 0.999), weight_decay=args.deepmil_reg)
        else:
            optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=args.deepmil_lr, betas=(0.9, 0.999), weight_decay=args.deepmil_reg)
    else:
        print(f"{args.classification_head} has not been implemented as classification head")
        raise NotImplementedError

    model = model.to(args.device)
    print(model)
    

    ### ============= GET THE CORRECT DATA LOADERS =============  ###

    if args.precompute_features or not args.use_precomputed_features:
        # If we precompute features, we need an image loader
        # If we use precomputed features, we don't need an image loader
        # If we don't use any preocomputed features, which happens if we finetune, we do want an image loader 

        assert not (args.precompute_features and args.use_precomputed_features), "Ambiguous config. Precompute features or use precomputed features?"

        drop_last = not (args.precompute_features and not args.precompute_features_in_memory) # if we precompute features, but NOT in memory, do not drop last

        if args.dataset in ['msi-tcga','basis']:
            # For msi-tcga, we have pre-split everything
            train_sampler=None
            val_sampler=None
    
        else:
            train_indices, val_indices = get_train_val_indices(train_dataset, val_split=args.validation_split)
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)  
            val_dataset = train_dataset
            # for non-msi-tcga, we have to split, still

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.logistic_batch_size,
            drop_last=drop_last,
            num_workers=args.workers,
            sampler=train_sampler
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
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
     

    ### ============ Precompute and/or load precomputed features ============= ###

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

            print("### Aggregating saved feature vectors into patient-level tensors ###")
            print("### Aggregating for train...")
            aggregate_patient_vectors(args, root_dir=args.path_to_msi_data, append_with=f'_{run_id}', grid=True, data=train_dataset.labels)
            print("### Aggregating for val...")
            aggregate_patient_vectors(args, root_dir=args.path_to_msi_data, append_with=f'_{run_id}', grid=True, data=val_dataset.labels)
            print("### Aggregating for test...")
            aggregate_patient_vectors(args, root_dir=args.path_to_msi_data, append_with=f'_{run_id}', grid=True, data=test_dataset.labels)
            
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

    if 'linear' in args.classification_head:
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=arr_train_loader.dataset.get_class_balance().float().to(args.device)) #TODO add class balance

    val_losses = []
    val_roc = []
    global_step = 0
    for epoch in range(args.logistic_epochs):
        args.current_epoch = epoch
        loss_epoch, accuracy_epoch, val_losses, val_roc, global_step = train(args, arr_train_loader, arr_val_loader, extractor, model, criterion, optimizer, val_losses, val_roc, global_step, epoch, writer)
        writer.add_scalar("loss/train", loss_epoch / len(arr_train_loader), epoch)


    ### ============= TESTING =============  ###

    if args.logistic_epochs > 0:
        # If we have run epochs in the current run, we will take the best model from the current run
        if not 'best_model_evaluation' in vars(args).keys():
            args.best_model_evaluation = 'auc'

        if args.best_model_evaluation == 'auc' or args.best_model_evaluation == 'r2': # r2 for linear regression, which is saved in the roc array
            # This seems like it would make most sense for the logistic regression (tile-level prediction & majority vote)
            best_model_num = np.argmax(val_roc)
        elif args.best_model_evaluation == 'loss':
            # This is probably most sensible for patient-level prediction methods like deepmil
            best_model_num = np.argmin(val_losses)
        
        best_model_epoch = best_model_num * args.evaluate_every + args.evaluate_every

        print(f'Validation ROCs: {val_roc}\nValidation loss: {val_losses}.\nBest performance by model @ epoch # {best_model_epoch}')

        if extractor:
            extractor_fp = os.path.join(args.out_dir, "extractor_checkpoint_{}.tar".format(best_model_epoch))
            extractor.load_state_dict(torch.load(extractor_fp, map_location=args.device.type))
        model_fp = os.path.join(args.out_dir, "classifier_checkpoint_{}.tar".format(best_model_epoch))
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))

    else:
        # clarifying that we have not trained at all
        best_model_epoch=epoch=0

        # If we haven't run any epochs, we won't do anything, because we've already loaded the model before...

        # if args.reload_model:
        #     if extractor:
        #         extractor_fp = os.path.join(args.model_path, "extractor_checkpoint_{}.tar".format(args.epoch_num))
        #         extractor.load_state_dict(torch.load(extractor_fp, map_location=args.device.type))
        #     model_fp = os.path.join(args.model_path, "classifier_checkpoint_{}.tar".format(args.epoch_num))
        #     model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
            
    loss_epoch, accuracy_epoch, labels, preds, patients, attentions, attention_gradients, img_names, subsample_indices, predicted_probs = validate(
        args, arr_test_loader, extractor, model, criterion, optimizer, final_test=True
    )


    ### ============= Compute final metrics and write them to tensorboard ============= ###

    writer.add_scalar("loss/test", loss_epoch / len(arr_test_loader))
    if 'linear' not in args.classification_head:
        # Logistic regression, using ROC AUC as metric
        final_data, rocauc = compute_roc_auc(args, patients, labels, preds, save_curve=True, epoch=best_model_epoch)
        writer.add_scalar("rocauc/test", rocauc)
        write_hparams(writer, args, {'hparam/rocauc': rocauc })
        _run.log_scalar('test/rocauc', rocauc, 0)
        print(
            f"======\n[Final test with best model]\t ROCAUC: {rocauc} \t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}"
        )
    else:
        # Linear regression, using R2 score as metric
        final_data, r2 = compute_r2(patients, labels, preds)
        writer.add_scalar("r2/test", r2)
        write_hparams(writer, args, {'hparam/r2': r2 })
        _run.log_scalar('test/r2', r2, 0)
        print(
            f"======\n[Final test with best model]\t R2: {r2} \t Loss: {loss_epoch / len(arr_test_loader)}\t"
        )


    ### ============= Save the predictions ============= ###

    humane_readable_time = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d-%H-%M-%S')
    print(f'This is the out dir {args.out_dir}')

    final_data.to_csv(f'{args.out_dir}/regression_output_epoch_{best_model_epoch}_{humane_readable_time}.csv')    

    if args.classification_head == 'deepmil':
        # This means we'll have loaded a stacked grid, we'll have returned subsample indices, we'll have meaningful attentions.
        # Now save it for easy visualization..
        save_attention(args, img_names, attentions, attention_gradients, subsample_indices, args.out_dir, humane_readable_time, best_model_epoch)
    elif args.classification_head == 'logistic':
        save_probabilities(args, img_names, predicted_probs, labels, args.out_dir, humane_readable_time, best_model_epoch)

    
    

