import os
import torch
import torchvision.models as models
from modules import SimCLR, LARS, BYOL


def load_model(args, loader, reload_model=False, model_type='simclr'):
    #TODO Loader is not used

    possible_non_simclr_models = {'imagenet-resnet18': models.resnet18, 'imagenet-resnet50': models.resnet50, 'imagenet-shufflenet-v1_x1_0': models.shufflenet_v2_x1_0}

    if model_type == 'simclr':
        if args.feature_learning == "unsupervised":
            model = SimCLR(args)        
        elif args.feature_learning == "supervised":
            args.normalize = False      # we get class outputs, so no need for a normalize
            args.projection_dim = 2     # num_classes. we are mostly looking at binary classification problems
            model = SimCLR(args)
        else:
            raise NotImplementedError

        if reload_model:
            model_fp = os.path.join(
                args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
            )
            print(f'### Loading model from: {model_fp} ###')
            model.load_state_dict(torch.load(model_fp, map_location=args.device.type))

        model = model.to(args.device)

    elif 'imagenet' in model_type:
        assert (model_type in possible_non_simclr_models), f"{model_type} is not supported. Please choose a model from {possible_non_simclr_models.keys()}"
        model = possible_non_simclr_models[model_type](pretrained=True)
        model.fc = torch.nn.Identity()
    elif model_type == 'byol':
        #TODO make more flexible
        #TODO Add several resnets
        #TODO Add possibility of loading previous models
        backbone = models.resnet18(pretrained=False)
        model = BYOL(
            net=backbone,
            image_size = 225,
            hidden_layer='avgpool'
        )
    else:
        raise NotImplementedError

    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    elif args.optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Install the apex package from https://www.github.com/nvidia/apex to use fp16 for training"
            )

        print("### USING FP16 ###")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    if model_type=='byol':
        return model, optimizer, scheduler, backbone
    else:
        return model, optimizer, scheduler


def save_model(args, model, optimizer, prepend=''):

    out = os.path.join(args.out_dir, "{}checkpoint_{}.tar".format(prepend, args.current_epoch))

    print(f'Saving model to {out}')

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)
