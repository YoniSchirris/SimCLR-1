import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np

from experiment import ex
from model import load_model
from utils import post_config_hook

from modules import LogisticRegression
from modules.deepmil import Attention
from modules.transformations import TransformsSimCLR

from msidata.dataset_msi import PreProcessedMSIDataset as dataset_msi

import pandas as pd
import time
import datetime



def inference(loader, context_model, device):
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


def get_features(context_model, train_loader, test_loader, device):
    train_X, train_y, train_patients, train_imgs = inference(train_loader, context_model, device)
    test_X, test_y, test_patients, test_imgs = inference(test_loader, context_model, device)
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

        if args.classification_head == 'logistic':
            output = model(x)
            loss = criterion(output, y)
            predicted = output.argmax(1)
            acc = (predicted == y).sum().item() / y.size(0)
            loss_epoch += loss.item()

        elif args.classification_head == 'deepmil':
            Y_prob, Y_hat, A = model.forward(x)
            loss, _ = model.calculate_objective(data, bag_label, Y_prob, A)
            train_loss += loss.data[0]
            loss_epoch += train_loss
            error, _ = model.calculate_classification_error(data, bag_label, Y_hat)
            acc = 1. - error
        
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def test(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0

    labels = []
    preds = [] 
    patients = []
    img_names = []

    model.eval()
    for step, (x, y, patient) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)

        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

        labels += y.cpu().tolist()
        preds += predicted.cpu().tolist()
        patients += patient.cpu().tolist()


    return loss_epoch, accuracy_epoch, labels, preds, patients

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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    simclr_model, _, _ = load_model(args, train_loader, reload_model=True, model_type=args.logistic_extractor)
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    ## Logistic Regression
    # n_classes = 10  # stl-10
    n_classes = 2  # MSI VS MSS
    
    if args.logistic_extractor == 'simclr':
        n_features = simclr_model.encoder.fc.in_features
    else:
        n_features = {'imagenet-resnet18': 512, 'imagenet-resnet50': 2048, 'imagenet-simclr_v1_x1_0': 1024}[args.logistic_extractor]

    if args.classification_head == 'logistic':
        model = LogisticRegression(n_features, n_classes)
    elif args.classification_head == 'deepmil':
        model = Attention()
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y, train_patients, train_imgs, test_patients, test_imgs) = get_features(
        simclr_model, train_loader, test_loader, args.device
    )

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.logistic_batch_size, train_patients, train_imgs, test_patients, test_imgs
    )

    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(
            args, arr_train_loader, simclr_model, model, criterion, optimizer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}"
        )

    # final testing
    loss_epoch, accuracy_epoch, labels, preds, patients = test(
        args, arr_test_loader, simclr_model, model, criterion, optimizer
    )

    final_data = pd.DataFrame(data={'patient': patients, 'labels': labels, 'preds': preds})

    humane_readable_time = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d-%H-%M-%S')

    print(f'This is the out dir {args.out_dir}')
    final_data.to_csv(f'{args.out_dir}/regression_output_{humane_readable_time}.csv')

    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}"
    )

