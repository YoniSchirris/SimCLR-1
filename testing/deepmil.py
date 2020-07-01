import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from experiment import ex
from model import load_model
from utils import post_config_hook

from modules import LogisticRegression
from modules.deepmil import Attention
from modules.transformations import TransformsSimCLR
from modules.sync_batchnorm import convert_model


from msidata.dataset_msi_features_with_patients import PreProcessedMSIFeatureDataset as dataset_msi_features

import pandas as pd
import time
import datetime
import time
import os

def train(args, loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    accuracy_epoch = 0
    t5=time.time()
    for step, data in enumerate(loader):
        print(f'Getting an item from enumerate loader takes {time.time()-t5:.2f} seconds')
        t1 = time.time()
        optimizer.zero_grad()

        x = data[0]
        Y = data[1]
        patients = data[2]
        
        x = x.to(args.device)
        Y = Y.to(args.device)

        if args.classification_head == 'logistic':
            output = model(x)
            loss = criterion(output, Y)
            predicted = output.argmax(1)
            acc = (predicted == Y).sum().item() / y.size(0)
            loss_epoch += loss.item()

        elif args.classification_head == 'deepmil':

            assert(Y.max() == Y.min()), "A single patient has different labels, which should not be possible"
            Y = Y.max() # The bag has a single label

            assert(len(set(patients)) == 1), f"We are loading several patients instead of just one. Check your dataloader. \nPatients: {patients}"
            patient = set(patients)

            t2 = time.time()
            Y_out, Y_hat, A = model.forward(x)
            print(f'Forward took {time.time()-t2:.2f} seconds')

            loss = criterion(input=Y_out, target=Y.unsqueeze(0))      # use toch crossentropy loss instead of self-engineered loss

            error, _ = model.calculate_classification_error(Y, Y_hat)
            acc = 1. - error

            train_loss = loss.item()
            loss_epoch += train_loss
            args.global_step+=1

        accuracy_epoch += acc

        t3 = time.time()
        loss.backward()
        print(f'Gradient calculation takes {time.time()-t3:.2f} seconds')

        t4 = time.time()
        optimizer.step()
        print(f'Optimizer takes {time.time()-t4:.2f} seconds')

        print(f'Total: Takes {time.time()-t1:.2f} seconds')
        t5 = time.time()

        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def save_model(args, model, optimizer):
    out = os.path.join(args.out_dir, "checkpoint_{}.tar".format(args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)



def test(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0

    labels = []
    preds = [] 
    saved_patients = []
    saved_input_paths = []

    model.eval()

    for step, data in enumerate(loader):
        x = data[0]
        Y = data[1]
        patients = data[2]
        input_paths = data[3]

        model.zero_grad()



        x = x.to(args.device)
        Y = Y.to(args.device)

        if args.classification_head == 'logistic':
            output = model(x)

            loss = criterion(output, Y)

            predicted = output.argmax(1)
            acc = (predicted == Y).sum().item() / Y.size(0)
            accuracy_epoch += acc

            loss_epoch += loss.item()

            labels += Y.cpu().tolist()
            preds += predicted.cpu().tolist()
            saved_patients += patients.cpu().tolist()
            saved_input_paths += input_paths

        elif args.classification_head == 'deepmil':
            assert(Y.max() == Y.min()), "A single patient has different labels, which should not be possible"
            Y = Y.max() # The bag has a single label

            assert(len(set(patients)) == 1), f"We are loading several patients instead of just one. Check your dataloader. Patients: {patients}. Input: {x}. Labels: {Y}"
            patient = patients[0] # Since all are the same, we can pick the first element to be the patient
            patient=patient[0] # due to https://github.com/pytorch/pytorch/issues/6893
            input_paths = [path[0] for path in input_paths]

            output = model(x)
            Y_out, Y_hat, A = model.forward(x)
            loss = criterion(input=Y_out, target=Y.unsqueeze(0))      # use toch crossentropy loss instead of self-engineered loss

            train_loss = loss.item()
            
            error, _ = model.calculate_classification_error(Y, Y_hat)
            acc = 1. - error

            loss_epoch += train_loss
            accuracy_epoch += acc

            Y_prob = Y_out.float().softmax(dim=1)[:,1].cpu().item() # We get probabilities, and get the probability for MSI for all in batch

            labels.append(Y.item())   # Y is a single label
            preds.append(Y_prob) 
            saved_patients.append(patient)  # We asserted that we have a single patient
            saved_input_paths += input_paths    # This is fairly useless now, as we can't combine them. But they will be useful for backpropping the MSIness label

            # TODO Some way of getting attention per tile here

        else:
            raise NotImplementedError

    return loss_epoch, accuracy_epoch, labels, preds, saved_patients

@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.n_gpu = torch.cuda.device_count()

    root = "./datasets"

    if args.dataset == "msi" and args.classification_head == 'deepmil':
        sampling_strategy = 'patient'
    else:
        raise NotImplementedError

    if args.dataset == "msi":
        train_dataset = dataset_msi_features(
            root_dir            =   args.path_to_msi_data, 
            sampling_strategy   =   sampling_strategy,
            data_fraction       =   args.data_testing_train_fraction,
            device              =   args.device,
            balance_classes     =   args.data_testing_balance_training_data
            )
        test_dataset = dataset_msi_features(
            root_dir            =   args.path_to_test_msi_data, 
            sampling_strategy   =   sampling_strategy,
            data_fraction       =   args.data_testing_test_fraction,
            device              =   args.device,
            balance_classes     =   False   # I'd rather have all data, and subsample after predictions have been made
            )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,   # taking 1 patient each. getting all patient's tiles
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    # simclr_model, _, _ = load_model(args, train_loader, reload_model=True)
    # simclr_model = simclr_model.to(args.device)
    # simclr_model.eval()

    ## Logistic Regression
    # n_classes = 10  # stl-10
    n_classes = 2  # MSI VS MSS

    if args.classification_head == 'logistic':
        model = LogisticRegression(simclr_model.n_features, n_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
    elif args.classification_head == 'deepmil':
        model = Attention()
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.clsfc_lr, betas=(0.9, 0.999), weight_decay=args.clsfc_reg)   #---- optimizer from deepMIL

        # Test other optimizer..
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        class_distribution = train_dataset.get_class_distribution()
        print(f'Class distribution for MSS, MSI is {class_distribution}')
        class_weights = class_distribution.max() / class_distribution ## Scales e.g. [0.15,0.85] to [5.66, 1]
        print(f'Setting CELoss weights to {class_weights}')
        criterion = torch.nn.CrossEntropyLoss(weight=class_distribution.to(args.device))

    model = model.to(args.device)

    
    

    # No multi-gpu support for now
    # print(f"Using {args.n_gpu} GPUs")
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    #     model = convert_model(model)
    #     model = model.to(args.device)
    #     args.batch_size *= args.n_gpu
    #     print(f"Using a total batch size of {args.batch_size} spread over the GPUs now")

    print(model)

    tb_dir = os.path.join(args.out_dir, args.classification_head)
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    # Training loop
    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(
            args, train_loader, model, criterion, optimizer, writer
        )

        if epoch % 10 == 0:
            save_model(args, model, optimizer)

        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
        writer.add_scalar("Accuracy/train_epoch", accuracy_epoch / len(train_loader), epoch)

        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Misc/learning_rate", lr, epoch)
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}"
        )
        args.current_epoch +=1

    # final testing
    loss_epoch, accuracy_epoch, labels, preds, patients = test(
        args, test_loader, model, criterion, optimizer
    )

    
    

    writer.add_scalar("Loss/test", loss_epoch / len(test_loader), 1)
    writer.add_scalar("Accuracy/test", accuracy_epoch / len(test_loader), 1)

    final_data = pd.DataFrame(data={'patient': patients, 'labels': labels, 'preds': preds})

    humane_readable_time = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d-%H-%M-%S')

    
    output_csv = f'{args.out_dir}/regression_output_{humane_readable_time}.csv'
    print(f"Results saved to {output_csv}")
    final_data.to_csv(output_csv)

    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}"
    )

