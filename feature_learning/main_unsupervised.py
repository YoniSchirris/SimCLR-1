import os
import torch
import torchvision
import argparse
import time

from torch.utils.tensorboard import SummaryWriter

apex = False
try:
    from apex import amp
    apex = True
except ImportError:
    print(
        "Install the apex package from https://www.github.com/nvidia/apex to use fp16 for training"
    )

from model import load_model, save_model
from modules import NT_Xent
from modules.sync_batchnorm import convert_model
from modules.transformations import TransformsSimCLR
from utils import post_config_hook
from msidata.dataset_msi import PreProcessedMSIDataset as dataset_msi
from msidata.dataset_tcga_tiles import TiledTCGADataset as dataset_tcga

#### pass configuration
from experiment import ex

def train_simclr(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _, _, _) in enumerate(train_loader):
        print("Loading successfully...")
        print(x_i.shape)    

        optimizer.zero_grad()
        x_i = x_i.to(args.device)
        x_j = x_j.to(args.device)

        # positive pair, with encoding
        h_i, z_i = model(x_i)
        h_j, z_j = model(x_j)

        loss = criterion(z_i, z_j)

        if apex and args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        if step % 50 == 0:
            print(f"{time.ctime()} | Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
        loss_epoch += loss.item()
        args.global_step += 1
    
    return loss_epoch

def train_byol(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    print("Training BYOL!")
    for step, ((x_i, x_j), _, _, _) in enumerate(train_loader):
        # augmentations are done within the model
        # loss is computed within the model
        loss = model(image_one=x_i, image_two=x_j)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_moving_average()

        loss_epoch += loss.item()

        optimizer.step()

        if step % 50 == 0:
            print(f"{time.ctime()} | Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)

        args.global_step += 1

    return loss_epoch

def train(args, train_loader, model, criterion, optimizer, writer):

    if args.unsupervised_method == 'simclr':
        train_method = train_simclr
    elif args.unsupervised_method == 'byol':
        train_method = train_byol
    else:
        raise NotImplementedError

    loss_epoch = train_method(args, train_loader, model, criterion, optimizer, writer)
    
    return loss_epoch


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)

    if torch.cuda.is_available():
        print("--- USING GPU ---")
        args.device=torch.device("cuda:0")
    else:
        print("--- USING CPU ----")
        args.device=torch.device('cpu')
    args.n_gpu = torch.cuda.device_count()

    root = "./datasets"

    train_sampler = None

    transform = TransformsSimCLR(size=224) # if args.unsupervised_method == 'simclr' else None
    # When transform = None, the dataloader will retrieve only a single image that is not transformed, as this will be done inside BYOL

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            root, split="unlabeled", download=True, transform=TransformsSimCLR(size=96)
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root, download=True, transform=TransformsSimCLR(size=32)
        )
    elif args.dataset == 'msi-kather':
        train_dataset = dataset_msi(root_dir=args.path_to_msi_data, transform=transform, data_fraction=args.data_pretrain_fraction)

    elif args.dataset == 'msi-tcga':    
        assert ('.csv' in args.path_to_msi_data), "Please provide the tcga .csv file in path_to_msi_data"
        assert (args.root_dir_for_tcga_tiles), "Please provide the root dir for the tcga tiles"
        train_dataset = dataset_tcga(csv_file=args.path_to_msi_data, root_dir=args.root_dir_for_tcga_tiles, transform=transform)            
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    if args.unsupervised_method=='byol':
        # Backbone is a reference to the network being used and updated. We should save the state of this network
        model, optimizer, scheduler, backbone = load_model(args, train_loader, reload_model=args.reload_model, model_type=args.unsupervised_method)
        
        # Criterion is defined within BYOL
        criterion = None
    else:
        model, optimizer, scheduler = load_model(args, train_loader, reload_model=args.reload_model, model_type=args.unsupervised_method)
        criterion = NT_Xent(args.batch_size, args.temperature, args.device)

    print(f"Using {args.n_gpu} GPUs")
    if 'use_multi_gpu' not in vars(args).keys():
        args.use_multi_gpu=False
    if args.n_gpu > 1 and args.use_multi_gpu:
        model = torch.nn.DataParallel(model)
        model = convert_model(model)
        #TODO Check the batch size.. are we only training with 32 total so 8 per GPU? That's veeeery few.

    model = model.to(args.device)

    print(model)

    tb_dir = os.path.join(args.out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)
    

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if scheduler:
            scheduler.step()

        if epoch % args.save_each_epochs == 0:
            if args.unsupervised_method == "simclr":
                # Save entire model
                save_model(args, model, optimizer)
            elif args.unsupervised_method == "byol":
                # Save only the resnet backbone
                save_model(args, backbone, optimizer)


        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
        writer.add_scalar("Misc/learning_rate", lr, epoch)
        print(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
        )
        args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)
