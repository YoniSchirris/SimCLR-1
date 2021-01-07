import os
import torch
import torchvision.models as models
from modules import SimCLR, LARS, BYOL, LogisticRegression
from modules.deepmil import Attention
from modules.deepmil import AttentionWithStd # TODO REMOVE IF WE DONT USE THIS ANYMORE...


def load_model(args, reload_model=False, model_type='simclr', prepend='', n_features=None, n_classes=None):

    possible_non_simclr_models = {'imagenet-resnet18': models.resnet18, 'imagenet-resnet50': models.resnet50, 'imagenet-shufflenetv2_x1_0': models.shufflenet_v2_x1_0}

    if 'reload_classifier' not in vars(args).keys():
        args.reload_classifier = False

    if args.reload_model:
        assert(isinstance(args.epoch_num, int) or args.epoch_num == 'best'), f"Don't know how to deal with the given epoch num: {args.epoch_num}. Only ints or 'best' are implemented"
        if args.epoch_num == 'best':
        # example regression output filename for the best model is 'regression_output_epoch_4_2020-10-07-17-25-03.csv'
        # However, it's saved as epoch+1. But the model is saved with "epoch"
            epoch_num = int([path for path in os.listdir(args.model_path) if 'regression_output' in path][0].split('_')[3]) - 1
        else:
            epoch_num = args.epoch_num
    if args.reload_classifier:
        assert(isinstance(args.classifier_epoch, int) or args.classifier_epoch == 'best'), f"Don't know how to deal with the given epoch num: {args.classifier_epoch}. Only ints or 'best' are implemented"
        if args.classifier_epoch == 'best':
        # example regression output filename for the best model is 'regression_output_epoch_4_2020-10-07-17-25-03.csv'
        # However, it's saved as epoch+1. But the model is saved with "epoch"
            classifier_epoch_num = int([path for path in os.listdir(args.classifier_path) if 'regression_output' in path][0].split('_')[3]) - 1
        else:
            classifier_epoch_num = args.classifier_epoch

    if model_type == 'simclr':
        if args.feature_learning == "unsupervised":
            model = SimCLR(args)        
        elif args.feature_learning == "supervised":
            #TODO This is not able to load the state dict due to the different weights than what it is trained for
            # Also, in the train script, we use a separate extractor and a separate classifier. 
            # So this is not really needed, we can just take it as is...

            args.normalize = False      # we get class outputs, so no need for a normalize
            args.projection_dim = 2     # num_classes. we are mostly looking at binary classification problems
            model = SimCLR(args)
        else:
            raise NotImplementedError

        if reload_model:
            model_fp = os.path.join(
                args.model_path, "checkpoint_{}.tar".format(epoch_num)
            )
            if not os.path.isfile(model_fp):
                print(f"### {model_fp} does not exist. We will try prepending with 'extractor'")
                model_fp = os.path.join(
                args.model_path, "extractor_checkpoint_{}.tar".format(epoch_num)
            )
            assert(os.path.isfile(model_fp)), f"### {model_fp} still doesn't exist. Please fix your hparams."
            print(f'### Loading extractor from: {model_fp} ###')
            model.load_state_dict(torch.load(model_fp, map_location=args.device.type))

        model = model.to(args.device)

    elif 'imagenet' in model_type:
        assert (model_type in possible_non_simclr_models), f"{model_type} is not supported. Please choose a model from {possible_non_simclr_models.keys()}"
        if not 'imagenet_not_pretrained' in vars(args).keys():
            args.imagenet_not_pretrained = False
        if args.imagenet_not_pretrained or args.reload_model:
            print("## Using a network from scratch!")
            pretrained = False
        else:
            print("## Using an imagenet pretrained network!")
            pretrained=True
        model = possible_non_simclr_models[model_type](pretrained=pretrained)
        model.fc = torch.nn.Identity()

        if args.reload_model:
            model_fp = os.path.join(
                args.model_path, "checkpoint_{}.tar".format(epoch_num)
            )
            if not os.path.isfile(model_fp):
                print(f"### {model_fp} does not exist. We will try prepending with 'extractor'")
                model_fp = os.path.join(
                args.model_path, "extractor_checkpoint_{}.tar".format(epoch_num)
            )
            assert(os.path.isfile(model_fp)), f"### {model_fp} still doesn't exist. Please fix your hparams."
            print(f'### Loading imagenet-init and finetuned extractor from: {model_fp} ###')
            model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        model = model.to(args.device)

    elif model_type == 'byol':
        #TODO make more flexible
        #TODO Add several resnets
        #TODO Add possibility of loading previous models
        backbones = {
            "resnet18": models.resnet18(pretrained=False),
            "resnet50": models.resnet50(pretrained=False),
            "shufflenetv2_x1_0": models.shufflenet_v2_x1_0(pretrained=False)
        }

        backbone = backbones[args.resnet]
        n_features = backbone.fc.in_features

        if reload_model:
            backbone.fc = torch.nn.Identity()
            model_fp = os.path.join(
                args.model_path, "checkpoint_{}.tar".format(epoch_num)
            )
            if not os.path.isfile(model_fp):
                print(f"### {model_fp} does not exist. We will try prepending with 'extractor'")
                model_fp = os.path.join(
                args.model_path, "extractor_checkpoint_{}.tar".format(epoch_num)
            )
            assert(os.path.isfile(model_fp)), f"### {model_fp} still doesn't exist. Please fix your hparams."
            
            print(f'### Loading extractor from: {model_fp} ###')
            backbone.load_state_dict(torch.load(model_fp, map_location=args.device.type))
            
        model = BYOL(
            net=backbone,
            image_size = 224,
            hidden_layer=-2
        )

        model = model.to(args.device)
    elif model_type in ['deepmil', 'linear-deepmil', 'logistic', 'linear']:
        #TODO Currently only works for resnet18 backend
        if model_type in ['deepmil', 'linear-deepmil']:
            if not 'test_attention_stdev' in vars(args).keys():
                args.test_attention_stdev = False

            if 'test_remove_attention_bias' not in vars(args).keys():
                args.test_remove_attention_bias=False

            if args.test_remove_attention_bias:
                attention_bias=False
                print("*** Removing bias from Attention layer in DeepMIL!")
            else:
                attention_bias=True


            if args.test_attention_stdev:
                model = AttentionWithStd(hidden_dim=n_features, intermediate_hidden_dim=args.deepmil_intermediate_hidden, num_classes=n_classes)
            else:

                model = Attention(hidden_dim=n_features, intermediate_hidden_dim=args.deepmil_intermediate_hidden, num_classes=n_classes, attention_bias=attention_bias)
        else:
            model = LogisticRegression(n_features, n_classes)
        if reload_model:
            model_fp = os.path.join(
                args.classifier_path, f"classifier_checkpoint_{classifier_epoch_num}.tar"
            )
            print(f'### Loading classifier from: {model_fp} ###')
            model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    elif 'cnn' in model_type:
        if model_type == 'cnn-resnet18':
            print("==== Using resnet18 as classification head")
            model = torchvision.models.resnet18()
            model.fc = torch.nn.Linear(model.fc.in_features, n_classes) 
            # v--- this is the exact Conv2d line for conv1 found in the Resnet class
            # Instead, we manually set the in_channels to 512 instead of 3.
            # hardcoding out_channels=64 instead of model.inplanes because inplanes gets changed during initialization
            model.conv1 = torch.nn.Conv2d(512, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        else:
            # args.classification_head == 'cnn-densenet'
            print("==== Using densenet161 as classification head")
            model = torchvision.models.densenet161()
            model.classifier = torch.nn.Linear(model.classifier.in_features, n_classes, bias=True)
            model.features.conv0 = torch.nn.Conv2d(1024, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        if reload_model:
            model_fp = os.path.join(
                args.classifier_path, f"classifier_checkpoint_{classifier_epoch_num}.tar"
            )
            print(f'### Loading classifier from: {model_fp} ###')
            model.load_state_dict(torch.load(model_fp, map_location=args.device.type))

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
        return model, optimizer, scheduler, backbone, n_features
    else:
        return model, optimizer, scheduler


def save_model(args, model, optimizer, prepend=''):

    out = os.path.join(args.out_dir, "{}checkpoint_{}.tar".format(prepend, args.global_step))

    print(f'Saving model to {out}')

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)
