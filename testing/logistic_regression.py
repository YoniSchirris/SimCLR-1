import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np

from experiment import ex
from model import load_model, save_model
from utils import post_config_hook

from modules import LogisticRegression
from modules.deepmil import Attention
from modules.transformations import TransformsSimCLR
from modules.losses.focal_loss import FocalLoss

from msidata.dataset_msi import PreProcessedMSIDataset as dataset_msi
from msidata.save_feature_vectors import infer_and_save
from msidata.dataset_msi_features_with_patients import PreProcessedMSIFeatureDataset

import pandas as pd
import time
import datetime

from sklearn import metrics



def inference(args, loader, context_model, device):
    feature_vector = []
    labels_vector = []
    patients = []
    imgs = []
    for step, (x, y, patient, img_name) in enumerate(loader):
        x = x.to(device)

        # get encoding
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
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector, patients, imgs


def get_features(args, context_model, train_loader, test_loader, device):
    train_X, train_y, train_patients, train_imgs = inference(args, train_loader, context_model, device)
    test_X, test_y, test_patients, test_imgs = inference(args, test_loader, context_model, device)
    return train_X, train_y, test_X, test_y, train_patients, train_imgs, test_patients, test_imgs


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size, train_patients, train_imgs, test_patients, test_imgs):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train), torch.Tensor(train_patients)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test), torch.Tensor(test_patients)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, data in enumerate(loader):
        optimizer.zero_grad()

        x = data[0]
        y = data[1]

        x = x.to(args.device)
        y = y.to(args.device)

        if not (args.precompute_features or args.use_precomputed_features):
            if args.freeze_encoder:
                simclr_model.eval()
                with torch.no_grad():
                    out = simclr_model.forward(x)
     
            else:
                simclr_model.train()
                out = simclr_model.forward(x)
        
        
            if args.logistic_extractor=='simclr':
                # Simclr returns (h, z)
                h = out[0]
                x = h
            else:
                # Torchvision models return h
                h = out
                x = h
                # We name it x, since it's the input for the logistic regressors
    

        if args.classification_head == 'logistic':
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

        elif args.classification_head == 'deepmil':

            Y_prob, Y_hat, A = model.forward(x)
            
            loss = criterion(Y_prob, y)

            train_loss = loss.item()
            loss_epoch += train_loss
            error, _ = model.calculate_classification_error(y, Y_hat)
            acc = 1. - error
        
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def validate(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0

    labels = []
    preds = [] 
    patients = []
    img_names = []

    model.eval()

    for step, data in enumerate(loader):
        model.zero_grad()

        x = data[0]
        y = data[1]
        patient = data[2]

        x = x.to(args.device)
        y = y.to(args.device)

        if not (args.precompute_features or args.use_precomputed_features):
            simclr_model.eval()

            with torch.no_grad():
                out = simclr_model.forward(x)
     
            if args.logistic_extractor=='simclr':
                # Simclr returns (h, z)
                h = out[0]
                x = h
            else:
                # Torchvision models return h
                h = out
                x = h
                # We name it x, since it's the input for the logistic regressors

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

    train_dataset = PreProcessedMSIFeatureDataset(
        root_dir=args.path_to_msi_data, 
        transform=None, 
        data_fraction=1,
        sampling_strategy=sampling_strategy,
        append_img_path_with=f'_{run_id}',
        tensor_per_patient=args.load_patient_level_tensors
        
    )
    test_dataset = PreProcessedMSIFeatureDataset(
        root_dir=args.path_to_test_msi_data, 
        transform=None, 
        data_fraction=1,
        sampling_strategy=sampling_strategy,
        append_img_path_with=f'_{run_id}',
        tensor_per_patient=args.load_patient_level_tensors
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    return train_loader, test_loader


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = "./datasets"

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            root,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            root,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
    elif args.dataset == "msi":
        train_dataset = dataset_msi(
            root_dir=args.path_to_msi_data, 
            transform=TransformsSimCLR(size=224).test_transform, 
            data_fraction=args.data_testing_train_fraction)
        test_dataset = dataset_msi(
            root_dir=args.path_to_test_msi_data, 
            transform=TransformsSimCLR(size=224).test_transform, 
            data_fraction=args.data_testing_test_fraction)
    else:
        raise NotImplementedError



    if args.logistic_extractor == 'byol':
        # We get the rn18 backbone with the loaded state dict
        print("Loading BYOL model ... ")
        _, _, _, simclr_model, n_features = load_model(args, None, reload_model=args.reload_model, model_type=args.logistic_extractor)
    else:
        simclr_model, _, _ = load_model(args, None, reload_model=args.reload_model, model_type=args.logistic_extractor)
        simclr_model = simclr_model.to(args.device)

    if args.freeze_encoder:
        simclr_model.eval()
    else:
        simclr_model.train()

    ## Logistic Regression
    # n_classes = 10  # stl-10
    n_classes = 2  # MSI VS MSS
    
    if args.logistic_extractor == 'simclr':
        n_features = simclr_model.n_features
    elif args.logistic_extractor == 'byol':
        # We returned n_features in load_model()..
        pass
    else:
        n_features = {'imagenet-resnet18': 512, 'imagenet-resnet50': 2048, 'imagenet-simclr_v1_x1_0': 1024}[args.logistic_extractor]

    if args.classification_head == 'logistic':
        model = LogisticRegression(n_features, n_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    elif args.classification_head == 'deepmil':
        model = Attention(hidden_dim=n_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.deepmil_lr, betas=(0.9, 0.999), weight_decay=args.deepmil_reg)

        
    model = model.to(args.device)

    print(model)

    
    criterion = torch.nn.CrossEntropyLoss()


    if args.precompute_features or not args.use_precomputed_features:
        # If we precompute features, we need an image loader
        # If we use precomputed features, we don't need an image loader
        # If we don't use any preocomputed features, which happens if we finetune, we do want an image loader 

        assert not (args.precompute_features and args.use_precomputed_features), "Ambiguous config. Precompute features or use precomputed features?"

        drop_last = not (args.precompute_features and not args.precompute_features_in_memory) # if we precompute features, but NOT in memory, do not drop last

        #TODO ADD TRAIN_VAL SAMPLER

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.logistic_batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=args.workers,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.logistic_batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=args.workers,
        )         

    if args.precompute_features:
   
        if args.precompute_features_in_memory:
            print("### Creating features from pre-trained context model ###")
            (train_X, train_y, test_X, test_y, train_patients, train_imgs, test_patients, test_imgs) = get_features(
                args, simclr_model, train_loader, test_loader, args.device
            )

            arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
                train_X, train_y, test_X, test_y, args.logistic_batch_size, train_patients, train_imgs, test_patients, test_imgs
            )
        else:
            print("### Creating and saving features from pre-trained context model ###")
            assert args.data_testing_train_fraction == 1 and args.data_testing_test_fraction == 1, "Bugs might occur when we do not save feature vectors for all data due to sampling issues"

            run_id = args.out_dir.split('/')[-1] # "./logs/pretrain/<id>"

            # This overwrites any other saved feature vectors we have. That means that we can NOT run several scripts at the same time..

            infer_and_save(loader=train_loader, context_model=simclr_model, device=args.device, append_with=f'_{run_id}', model_type=args.logistic_extractor)
            infer_and_save(loader=test_loader, context_model=simclr_model, device=args.device, append_with=f'_{run_id}', model_type=args.logistic_extractor)
            
            # Overwriting previous variable names to reduce memory load
            train_loader, test_loader = get_precomputed_dataloader(args, run_id)

            arr_train_loader, arr_test_loader = train_loader, test_loader

    elif args.use_precomputed_features:
        
        assert (args.use_precomputed_features_id), 'Please set the run ID of the features you want to use'
        print(f"Removing SIMCLR model from memory, as we use precomputed features..")
        del simclr_model
        simclr_model = None
        arr_train_loader, arr_test_loader = get_precomputed_dataloader(args, args.use_precomputed_features_id)
    else:
        arr_train_loader, arr_test_loader = train_loader, test_loader


    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(
            args, arr_train_loader, simclr_model, model, criterion, optimizer
        )
        print(
            f"{datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S')} | Epoch [{epoch+1}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
        )
        if (epoch+1) % args.save_logistic_model_each_epochs == 0:
            args.current_epoch = epoch+1
            if simclr_model:
                save_model(args, simclr_model, None, prepend='extractor_')
            save_model(args, model, None, prepend='classifier_')
                

            # Test every x epochs
            loss_epoch, accuracy_epoch, labels, preds, patients = validate(
                args, arr_test_loader, simclr_model, model, criterion, optimizer
            )

            final_data = pd.DataFrame(data={'patient': patients, 'labels': labels, 'preds': preds})

            humane_readable_time = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d-%H-%M-%S')

            print(f'This is the out dir {args.out_dir}')
            final_data.to_csv(f'{args.out_dir}/regression_output_epoch_{epoch+1}_{humane_readable_time}.csv')

            dfgroup = final_data.groupby(['patient']).mean()
            labels = dfgroup['labels'].values
            preds = dfgroup['preds'].values
            rocauc=metrics.roc_auc_score(y_true=labels, y_score=preds)


            print(
                f"[TEST EPOCH {epoch+1}]\t ROCAUC: {rocauc} \t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}"
            )

