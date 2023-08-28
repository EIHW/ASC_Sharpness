from sincnet import (
    SincNet,
    MLP
)
from models import (
    Cnn10,
    Cnn14,
    create_ResNet50_model,
    ModifiedEfficientNet,
    ASTModel
)
from datasets import (
    CachedDataset,
    WavDataset
)
from utils import (
    disaggregated_evaluation,
    evaluate_categorical,
    transfer_features,
    LabelEncoder,
    get_output_dim,
    get_df_from_dataset,
    GrayscaleToRGB
)
from transformers import (
    ASTFeatureExtractor
)


#from ml_utils import get_sharpness
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import argparse
import audtorch
import numpy as np
import os
import pandas as pd
import random
import shutil
import torch
import copy
import tqdm
import yaml
from KFACPytorch import KFACOptimizer, EKFACOptimizer
from sam import SAM
#from calculate_different_sharpness_values import calculate_sharpness
from gradient_descent_the_ultimate_optimizer.gdtuo import ModuleWrapper, NoOpOptimizer
from torchinfo import summary

import warnings
# Ignore Using backward() UserWarning of gdtuo
warnings.filterwarnings(category=UserWarning, action="ignore")


def fix_index(df, root):
    df.reset_index(inplace=True)
    df['filename'] = df['filename'].apply(
        lambda x: os.path.join(root, x))
    df.set_index('filename', inplace=True)
    return df


def replace_file_path(df: pd.DataFrame, col, new_path):
    def repl(st):
        return new_path + "/" + st.split("/")[-1]
    df[col] = df[col].apply(repl)
    return df


class Model(torch.nn.Module):
    def __init__(self, cnn, mlp_1, mlp_2, wlen, wshift):
        super().__init__()
        self.cnn = cnn
        self.mlp_1 = mlp_1
        self.mlp_2 = mlp_2
        self.wlen = wlen
        self.wshift = wshift
        self.output_dim = self.mlp_2.fc_lay[-1]

    def forward(self, x):
        # x = x.transpose(1, 2)
        if not self.training:
            x = x.unfold(1, self.wlen, self.wshift).squeeze(0)
        out = self.mlp_2(self.mlp_1(self.cnn(x)))
        if not self.training:
            out = out.mean(0, keepdim=True)
        return out


def train_step_gdtuo(model,
                     mw: ModuleWrapper,
                     criterion,
                     features,
                     targets,
                     device,
                     clip_net=1.,
                     clip_opt=1.,
                     ):
    # * Train Step for GDTUO Optimizer using ModelWrapper
    # ? Reference: https://github.com/kach/gradient-descent-the-ultimate-optimizer
    mw.begin()
    output = mw.forward(transfer_features(features, device))
    targets = targets.to(device)
    loss = criterion(output, targets)
    mw.zero_grad()
    loss.backward(create_graph=True)  # important! use create_graph=True
    # * GDTUO needs gradient clipping, a lot of stacked optimizers cause HUUUGE gradients!
    if clip_net is not None:
        for param in mw.all_params_with_gradients:
            _clip = torch.ones_like(param.grad) * clip_net
            param.grad = torch.minimum(param.grad, _clip)
            param.grad = torch.maximum(param.grad, -_clip)
    if clip_opt is not None:
        opt = mw.optimizer
        while not isinstance(opt, NoOpOptimizer):
            for param in opt.parameters.values():
                _clip = torch.ones_like(param.grad) * clip_opt
                param.grad = torch.minimum(param.grad, _clip)
                param.grad = torch.maximum(param.grad, -_clip)
            opt = opt.optimizer
    mw.step()
    _loss = loss.item()
    # * GDTUO leaks memory, so it needs to be dealt with manually!
    opt = mw
    while not isinstance(opt, NoOpOptimizer):
        if hasattr(opt, "all_params_with_gradients"):
            for param in opt.all_params_with_gradients:
                param.grad = None
            opt.all_params_with_gradients.clear()
        opt = opt.optimizer
    torch.cuda.empty_cache()
    return _loss


def train_step_kfac(model, optimizer, criterion, features, targets, device, _epoch, _batch):
    # * Train Step for (E)KFAC Optimizer
    # ? Reference: https://github.com/alecwangcq/KFAC-Pytorch    
    optimizer.zero_grad()
    output = model(transfer_features(features, device))
    targets = targets.to(device)
    loss = criterion(output, targets)
    if optimizer.steps % optimizer.TCov == 0:
        # compute true fisher
        optimizer.acc_stats = True
        with torch.no_grad():
            sampled_y = torch.multinomial(torch.nn.functional.softmax(output.cpu().data, dim=1),
                                          1).squeeze().cuda()
        loss_sample = criterion(output, sampled_y)
        loss_sample.backward(retain_graph=True)
        optimizer.acc_stats = False
        optimizer.zero_grad()  # clear the gradient for computing true-fisher.
    loss.backward()
    optimizer.step()
    _loss = loss.item()
    return _loss


def train_step_normal(model, optimizer, criterion, features, targets, device):
    # * Train Step for Torch Base Optimizers
    # print("-"*50)
    # TODO: Remove this part. It's only for testing.
    # sharp = get_sharpness(mode, train_dataset)
    # print("Sharpness: ", sharp)
    # print("Feature Shapes: ", features.shape)
    output = model(transfer_features(features, device))
    targets = targets.to(device)
    loss = criterion(output, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _loss = loss.item()
    return _loss

def train_step_SAM(model, optimizer, criterion, features, targets, device):
    # * Train Step for SAM optimizer
    output = model(transfer_features(features, device))
    targets = targets.to(device)
    # first forward-backward pass
    loss = criterion(output, targets)  # use this loss for any training statistics
    loss.backward()
    optimizer.first_step(zero_grad=True)
    
    # second forward-backward pass
    output = model(transfer_features(features, device))
    targets = targets.to(device)
    loss = criterion(output, targets)  # make sure to do a full forward pass
    loss.backward()
    optimizer.second_step(zero_grad=True)
    _loss = loss.item()
    return _loss


def run_training(args):
    def _get_device_multiprocessing(device):
        torch.cuda.set_device(torch.cuda.device(device))
        return "cuda:"+str(torch.cuda.current_device())
    args.device = args.device if isinstance(
        args.device, str) else _get_device_multiprocessing(args.device)
    torch.manual_seed(args.seed)
    gen_seed = torch.Generator().manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = args.device
    epochs = args.epochs
    experiment_folder = args.results_root
    os.makedirs(experiment_folder, exist_ok=True)
    ### DCASE
    if args.dataset == 'DCASE2020':
        def get_scene_category(x):
            if x in [
                'airport',
                'shopping_mall',
                'metro_station'
            ]:
                return 'indoor'
            elif x in [
                'park',
                'public_square',
                'street_pedestrian',
                'street_traffic'
            ]:
                return 'outdoor'
            elif x in [
                'bus',
                'metro',
                'tram'
            ]:
                return 'transportation'
            else:
                raise NotImplementedError(f'{x} not supported.')
        #print("-----------------hi---------------")
        # old
        # df_train = pd.read_csv(
        #     os.path.join(
        #         args.data_root,
        #         'evaluation_setup',
        #         'fold1_train.csv'
        #     ), sep='\t').set_index('filename')
        
        # new
        df_train = pd.read_csv(
            os.path.join(
                args.data_root,
                'evaluation_setup',
                'fold1_train.csv'
            ), sep='\t')
        df_train['filename'] = df_train['filename'].apply(lambda x: os.path.join(args.data_root, x))
        df_train = df_train.set_index('filename')


        df_train['scene_category'] = df_train['scene_label'].apply(
            get_scene_category)
        df_train['city'] = [
            os.path.basename(x).split('-')[1]
            for x in df_train.index.get_level_values('filename')
        ]
        df_train['device'] = [
            os.path.basename(x).split('-')[-1].split('.')[0]
            for x in df_train.index.get_level_values('filename')
        ]
        
        df_dev = pd.read_csv(
            os.path.join(
                args.data_root,
                'evaluation_setup',
                'fold1_evaluate.csv'
            ), sep='\t')
        df_dev['filename'] = df_dev['filename'].apply(lambda x: os.path.join(args.data_root, x))
        df_dev = df_dev.set_index('filename')
        df_dev['scene_category'] = df_dev['scene_label'].apply(get_scene_category)
        df_dev['city'] = [
            os.path.basename(x).split('-')[1]
            for x in df_dev.index.get_level_values('filename')
        ]
        df_dev['device'] = [
            os.path.basename(x).split('-')[-1].split('.')[0]
            for x in df_dev.index.get_level_values('filename')
        ]
        
        df_test = pd.read_csv(
            os.path.join(
                args.data_root,
                'evaluation_setup',
                'fold1_evaluate.csv'
            ), sep='\t')
        df_test['filename'] = df_test['filename'].apply(lambda x: os.path.join(args.data_root, x))
        df_test = df_test.set_index('filename')

        df_test['scene_category'] = df_test['scene_label'].apply(
            get_scene_category)
        df_test['city'] = [
            os.path.basename(x).split('-')[1]
            for x in df_test.index.get_level_values('filename')
        ]
        df_test['device'] = [
            os.path.basename(x).split('-')[-1].split('.')[0]
            for x in df_test.index.get_level_values('filename')
        ]

        if args.category is not None:
            df_train = df_train.loc[df_train['scene_category'] == args.category]
            df_dev = df_dev.loc[df_dev['scene_category'] == args.category]
            df_test = df_test.loc[df_test['scene_category'] == args.category]

        if args.exclude_cities != "None":
            df_train = df_train.loc[~df_train["city"].isin(args.exclude_cities)]

        n_classes = len(df_train['scene_label'].unique())
        encoder = LabelEncoder(
            list(df_train['scene_label'].unique()))
        # old
        #features = pd.read_csv(args.features).set_index('filename')
        # new
        features = pd.read_csv(args.features)
        features['filename'] = features['filename'].apply(lambda x: os.path.join(args.data_root, x))
        features = features.set_index('filename')


        # * custom feature path support
        if args.custom_feature_path is not None:
            features = replace_file_path(
                features, "features", args.custom_feature_path)

        db_args = {
            'features': features,
            'target_column': 'scene_label',
            'target_transform': encoder.encode,
            'feature_dir': args.feature_dir
        }
        
        if args.approach == 'cnn14':
            model = Cnn14(
                output_dim=n_classes
            )
            print("Pretrained: ", args.pretrained)
            if args.pretrained:
                # model_old = copy.deepcopy(model)
                # sd = model_old.state_dict()
                # for params in sd:
                #     print(params, sd[params])   
                checkpoint = torch.load(args.pretrained_dir + "Cnn14_16k_mAP=0.438.pth", map_location=torch.device(args.device))
                state_dict = checkpoint['model']
                model.load_state_dict(state_dict, strict=False)
            db_class = CachedDataset
            model.to_yaml(os.path.join(experiment_folder, 'model.yaml'))
            criterion = torch.nn.CrossEntropyLoss()
            
        elif args.approach == 'cnn10':
            model = Cnn10(
                output_dim=n_classes
            )
            print("Pretrained: ", args.pretrained)
            if args.pretrained:
                # model_old = copy.deepcopy(model)
                # sd = model_old.state_dict()
                # for params in sd:
                #     print(params, sd[params])   
                checkpoint = torch.load(args.pretrained_dir + "Cnn10_mAP=0.380.pth", map_location=torch.device(args.device))
                state_dict = checkpoint['model']
                model.load_state_dict(state_dict, strict=False)
                # sd2 = model.state_dict()
                # for params in sd2:
                #     print(params, sd2[params])   
            db_class = CachedDataset
            model.to_yaml(os.path.join(experiment_folder, 'model.yaml'))
            criterion = torch.nn.CrossEntropyLoss()
        elif args.approach == 'resnet50':
            model = create_ResNet50_model(n_classes, pretrained=args.pretrained)
            db_class = CachedDataset
            db_args['transform'] = transforms.Compose([GrayscaleToRGB()])
            criterion = torch.nn.CrossEntropyLoss()

        elif args.approach.startswith("efficientnet"):
            model = ModifiedEfficientNet(n_classes, scaling_type=args.approach, pretrained=args.pretrained)
            db_class = CachedDataset
            # db_args['transform'] = transforms.Compose([GrayscaleToRGB(), transforms.ToTensor()])
            db_args['transform'] = transforms.Compose([GrayscaleToRGB()])
            # db_args['transform'] = torch.nn.Sequential(GrayscaleToRGB())
            #db_args['transform'] = torch.nn.Sequential(torchvision.transforms.v2.Grayscale(num_output_channels=3))

            # model.to_yaml(os.path.join(experiment_folder, 'model.yaml'))
            criterion = torch.nn.CrossEntropyLoss()
        elif args.approach == 'ast':
            model = ASTModel(
                num_hidden_layers=args.num_hidden_layers
            )
            db_class = WavDataset
            criterion = torch.nn.CrossEntropyLoss()
            feature_extractor = ASTFeatureExtractor()

        elif args.approach == 'sincnet':
            with open('sincnet.yaml', 'r') as fp:
                options = yaml.load(fp, Loader=yaml.Loader)

            feature_config = options['windowing']
            wlen = int(feature_config['fs'] * feature_config['cw_len'] / 1000.00)
            wshift = int(feature_config['fs'] *
                        feature_config['cw_shift'] / 1000.00)

            cnn_config = options['cnn']
            cnn_config['input_dim'] = wlen
            cnn_config['fs'] = feature_config['fs']
            cnn = SincNet(cnn_config)

            mlp_1_config = options['dnn']
            mlp_1_config['input_dim'] = cnn.out_dim
            mlp_1 = MLP(mlp_1_config)

            mlp_2_config = options['class']
            mlp_2_config['input_dim'] = mlp_1_config['fc_lay'][-1]
            mlp_2 = MLP(mlp_2_config)
            model = Model(
                cnn,
                mlp_1,
                mlp_2,
                wlen,
                wshift
            )
            x = torch.rand(2, wlen)
            model.train()
            x = torch.rand(1, 44100)
            model.eval()
            # print("EVAL TEST:")
            # # print(model(x).shape)
            # print()
            with open(os.path.join(experiment_folder, 'sincnet.yaml'), 'w') as fp:
                yaml.dump(options, fp)
            db_class = WavDataset
            df_train = fix_index(df_train, args.data_root)
            df_dev = fix_index(df_dev, args.data_root)
            df_test = fix_index(df_test, args.data_root)
            db_args['transform'] = audtorch.transforms.RandomCrop(wlen)
            criterion = torch.nn.NLLLoss()

        train_dataset = db_class(
            df_train,
            **db_args
        )
        dev_dataset = db_class(
        df_dev,
        **db_args
        )

        test_dataset = db_class(
            df_test,
            **db_args
        )
    elif args.dataset == "CIFAR10":
        # Parameters for development set
        devel_percentage = 0.2
        if args.approach in ["cnn10", "cnn14"]:
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                torchvision.transforms.Resize((64,64))
                #torchvision.transforms.Resize((1001,64))
                ])
        else:
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        
        train_dev_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
        
        generator1 = torch.Generator().manual_seed(42)
        train_dataset, dev_dataset = torch.utils.data.random_split(train_dev_dataset, [1 - devel_percentage, devel_percentage], generator=generator1)
        
        
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

        df_dev = get_df_from_dataset(dev_dataset)
        df_test = get_df_from_dataset(test_dataset)
        
        encoder = LabelEncoder(
            list(test_dataset.class_to_idx.keys()))
        
        
        df_dev['label'] = df_dev['label'].apply(
                encoder.decode)
        df_test['label'] = df_test['label'].apply(
                encoder.decode)
                

        n_classes = len(test_dataset.class_to_idx.keys())
        input_channels = 3

        # So far only CNN14 and CNN10 are available
        if args.approach == 'cnn14':
            model = Cnn14(
                output_dim=n_classes,
                in_channels=input_channels
            )
            print("Pretrained: ", args.pretrained)
            if args.pretrained:
                # model_old = copy.deepcopy(model)
                # sd = model_old.state_dict()
                # for params in sd:
                #     print(params, sd[params])   
                # TODO: not implemented
                print("Not implemented...")
                # checkpoint = torch.load(args.pretrained_dir + "Cnn14_16k_mAP=0.438.pth", map_location=torch.device(args.device))
                # state_dict = checkpoint['model']
                # model.load_state_dict(state_dict, strict=False)
            db_class = CachedDataset
            model.to_yaml(os.path.join(experiment_folder, 'model.yaml'))
            criterion = torch.nn.CrossEntropyLoss()
            
        elif args.approach == 'cnn10':
            model = Cnn10(
                output_dim=n_classes,
                in_channels=input_channels
            )
            print("Pretrained: ", args.pretrained)
            if args.pretrained:
                # model_old = copy.deepcopy(model)
                # sd = model_old.state_dict()
                # for params in sd:
                #     print(params, sd[params])   
                # TODO: not implemented
                print("Not implemented...")
                # checkpoint = torch.load(args.pretrained_dir + "Cnn10_mAP=0.380.pth", map_location=torch.device(args.device))
                # state_dict = checkpoint['model']
                # model.load_state_dict(state_dict, strict=False)
                # sd2 = model.state_dict()
                # for params in sd2:
                #     print(params, sd2[params])   
            db_class = CachedDataset
            model.to_yaml(os.path.join(experiment_folder, 'model.yaml'))
            criterion = torch.nn.CrossEntropyLoss()
        elif args.approach == 'resnet50':
            model = create_ResNet50_model(n_classes, pretrained=args.pretrained)
        elif args.approach.startswith("efficientnet"):
            model = ModifiedEfficientNet(n_classes, scaling_type=args.approach, pretrained=args.pretrained)
            
        
    # Print a summary using torchinfo (uncomment for actual output)
    criterion = torch.nn.CrossEntropyLoss()        
    print(args.approach)
    if args.state is not None:
        initial_state = torch.load(args.state)
        model.load_state_dict(
            initial_state,
            strict=False
            )
    
    x, y = train_dataset[0]
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    summary(model=model, 
        input_size=(x.shape), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )
    # print("-" * 50)
    # personalized_plot_model(model)

    if args.approach == 'sincnet':
        db_args.pop('transform')
    
    # create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=4,
        generator=gen_seed
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=1 if args.approach == 'sincnet' else args.batch_size,
        num_workers=4,
        generator=gen_seed
    )
    

    # df_dev = pd.DataFrame(dev_dataset.dataset)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1 if args.approach == 'sincnet' else args.batch_size,
        num_workers=4,
        generator=gen_seed
    )

    accuracy_history = []
    uar_history = []
    f1_history = []
    train_loss_history = []
    valid_loss_history = []

    if not os.path.exists(os.path.join(experiment_folder, 'state.pth.tar')):

        encoder.to_yaml(os.path.join(experiment_folder, 'encoder.yaml'))
        with open(os.path.join(experiment_folder, 'hparams.yaml'), 'w') as fp:
            yaml.dump(vars(args), fp)

        writer = SummaryWriter(log_dir=os.path.join(experiment_folder, 'log'))

        torch.save(
            model.state_dict(),
            os.path.join(
                experiment_folder,
                'initial.pth.tar')
        )
        if isinstance(args.optimizer, str):
            optim = args.optimizer
            if args.optimizer == 'SGD':
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    momentum=0.9,
                    lr=args.learning_rate
                )
            elif args.optimizer == 'Adam':
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=args.learning_rate
                )
            elif args.optimizer == 'RMSprop':
                optimizer = torch.optim.RMSprop(
                    model.parameters(),
                    lr=args.learning_rate,
                    alpha=.95,
                    eps=1e-7
                )
        else:
            optimizer = args.optimizer.create(
                model, lr=args.learning_rate, device=args.device)

        if not "sheduler_wrapper" in args or args.sheduler_wrapper == None:
            sheduler = None
        elif isinstance(args.sheduler_wrapper, list):
            sheduler = []
            for sh in args.sheduler_wrapper:
                sheduler.append(sh.create(optimizer))
        else:
            sheduler = args.sheduler_wrapper.create(optimizer)

        

        max_metric = -1
        best_epoch = 0
        best_state = None
        best_results = None

        # accuracy_history = []
        # uar_history = []
        # f1_history = []
        # train_loss_history = []
        # valid_loss_history = []

        for epoch in range(epochs):
            model.to(device)
            model.train()
            epoch_folder = os.path.join(
                experiment_folder,
                f'Epoch_{epoch+1}'
            )
            os.makedirs(epoch_folder, exist_ok=True)

            if "train_timer" in args:
                args.train_timer.start()
            _loss_history = []
            for index, (features, targets) in tqdm.tqdm(
                enumerate(train_loader),
                desc=f'Epoch {epoch}',
                total=len(train_loader),
                disable=args.disable_progress_bar
            ):

                if (features != features).sum():
                    raise ValueError(features)

                if isinstance(optimizer, ModuleWrapper):
                    loss = train_step_gdtuo(
                        model, optimizer, criterion, features, targets, device)
                elif isinstance(optimizer, (KFACOptimizer, EKFACOptimizer)):
                    loss = train_step_kfac(
                        model, optimizer, criterion, features, targets, device, epoch+1, index+1)
                elif isinstance(optimizer, SAM):
                    loss = train_step_SAM(
                        model, optimizer, criterion, features, targets, device)
                else:
                    loss = train_step_normal(
                        model, optimizer, criterion, features, targets, device)
                if index % 50 == 0:
                    writer.add_scalar(
                        'Loss',
                        loss,
                        global_step=epoch * len(train_loader) + index
                    )
                _loss_history.append(loss)
                
            train_loss = sum(_loss_history)/len(_loss_history)
            # print(train_loss)
            if "train_timer" in args:
                args.train_timer.stop()

            if "valid_timer" in args:
                args.valid_timer.start()
            ##  Sharpness

            # sharpness_values = calculate_sharpness(model, device, train_loader, transfer_features, args.disable_progress_bar, criterion)
            # print(sharpness_values)
            



            # dev set evaluation
            results, _, predictions, outputs, valid_loss = evaluate_categorical(
                model,
                device,
                dev_loader,
                transfer_features,
                args.disable_progress_bar,
                criterion
            )
            results_df = pd.DataFrame(
                index=df_dev.index,
                data=predictions,
                columns=['predictions']
            )
            results_df['predictions'] = results_df['predictions'].apply(
                encoder.decode)
            results_df.reset_index().to_csv(os.path.join(epoch_folder, 'dev.csv'), index=False)
            np.save(os.path.join(epoch_folder, 'outputs.npy'), outputs)
        
            # print(results_df)
            if args.dataset == "DCASE2020":
                task = 'scene_label'
                stratify = ['scene_category', 'city', 'device']
            else:
                task = "label"
                stratify = []
            logging_results = disaggregated_evaluation(
                results_df,
                df_dev,
                task,
                stratify,
                'categorical'
            )

            with open(os.path.join(epoch_folder, 'dev.yaml'), 'w') as fp:
                yaml.dump(logging_results, fp)
            for metric in logging_results.keys():
                writer.add_scalars(
                    f'dev/{metric}',
                    logging_results[metric],
                    (epoch + 1) * len(train_loader)
                )

            torch.save(model.cpu().state_dict(), os.path.join(
                epoch_folder, 'state.pth.tar'))
            results["train_loss"] = train_loss
            results["val_loss"] = valid_loss
            
            print(f'Dev results at epoch {epoch+1}:\n{yaml.dump(results)}')
            # save accuracy metric
            accuracy_history.append(results["ACC"])
            uar_history.append(results["UAR"])
            f1_history.append(results["F1"])
            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)
            if results['ACC'] > max_metric:
                max_metric = results['ACC']
                best_epoch = epoch
                best_state = model.cpu().state_dict()
                best_results = results.copy()

            # plateau_scheduler.step(results['ACC'])
            if "valid_timer" in args:
                args.valid_timer.stop()
        train_results, _, _, _, train_loss = evaluate_categorical(
                model,
                device,
                train_loader,
                transfer_features,
                args.disable_progress_bar,
                criterion
            )
        print(f'Final Train results:\n {yaml.dump(train_results)}')
        print(f'Final Train loss:\n {yaml.dump(train_loss)}')
        #results["sharpness_value"] = sharpness_values
        print("Final Train results: ", train_results)
        print("Final Train loss: ", train_loss)

        # sharpness_values = calculate_sharpness(model, device, train_loader, transfer_features, args.disable_progress_bar, criterion)
        # print(f'Sharpness:\n{yaml.dump(results)}')
        #results["sharpness_value"] = sharpness_values
        # print("Sharpness Value: ", sharpness_values)
        print(
            f'Best dev results found at epoch {best_epoch+1}:\n{yaml.dump(best_results)}')
        best_results['Epoch'] = best_epoch + 1
        with open(os.path.join(experiment_folder, 'dev.yaml'), 'w') as fp:
            yaml.dump(best_results, fp)
        writer.close()
    else:
        best_state = torch.load(os.path.join(
            experiment_folder, 'state.pth.tar'))
        print('Training already run')
        epoch_folder = os.path.join(
                experiment_folder,
                f'Epoch_{epochs}'
            )
    # if args.dataset == "DCASE2020":
    
    print("saving to: ", os.path.join(experiment_folder, 'test_holistic.yaml'))
    if not os.path.exists(os.path.join(experiment_folder, 'test_holistic.yaml')):
    # if True:
        model.load_state_dict(best_state)
        test_results, targets, predictions, outputs, valid_loss = evaluate_categorical(
            model, device, test_loader, transfer_features, args.disable_progress_bar, criterion)
        print(f'Best test results:\n{yaml.dump(test_results)}')
        torch.save(best_state, os.path.join(
            experiment_folder, 'state.pth.tar'))
        np.save(os.path.join(experiment_folder, 'targets.npy'), targets)
        np.save(os.path.join(experiment_folder, 'outputs.npy'), outputs)
        np.save(os.path.join(experiment_folder, 'predictions.npy'), outputs)
        results_df = pd.DataFrame(
            index=df_test.index,
            data=predictions,
            columns=['predictions']
        )
        results_df['predictions'] = results_df['predictions'].apply(
            encoder.decode)
        results_df.reset_index().to_csv(os.path.join(epoch_folder, 'test.csv'), index=False)
        # print(results_df)
        # print(df_test)
        with open(os.path.join(experiment_folder, 'test.yaml'), 'w') as fp:
            yaml.dump(test_results, fp)
            if args.dataset == "DCASE2020":
                task = 'scene_label'
                stratify = ['scene_category', 'city', 'device']
            else:
                task = "label"
                stratify = []
            logging_results = disaggregated_evaluation(
                results_df,
                df_test,
                task,
                stratify,
                'categorical'
            )
            with open(os.path.join(experiment_folder, 'test_holistic.yaml'), 'w') as fp:
                yaml.dump(logging_results, fp)
    else:
        print('Evaluation already run')
        # in case we don't have any training the benchrunner doesn't make a lot of sense.

    return accuracy_history, uar_history, f1_history, train_loss_history, valid_loss_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DCASE-T1 Training')
    parser.add_argument(
        '--data-root',
        help='Path data has been extracted',
        required=True
    )
    parser.add_argument(
        '--results-root',
        help='Path where results are to be stored',
        required=True
    )
    parser.add_argument(
        '--features',
        help='Path to features',
        required=True
    )
    
    parser.add_argument(
        '--device',
        help='CUDA-enabled device to use for training',
        required=True
    )
    parser.add_argument(
        '--state',
        help='Optional initial state'
    )

    # TODO: is here efficientnet not supported anymore????
    parser.add_argument(
        '--approach',
        default='cnn10',
        choices=[
            'cnn14',
            'cnn10',
            'sincnet',
            'ast'
        ]
    )
    parser.add_argument(
        '--category',
        default=None,
        choices=[
            'indoor',
            'outdoor',
            'transportation',
            None
        ]
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=60
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0
    )
    parser.add_argument(
        '--optimizer',
        default='SGD'
    )
    parser.add_argument(
        '--feature_dir',
        default=''
    )

    parser.add_argument(
        '--custom-feature-path',
        help='Custom .npy location of features',
        required=False
    )
    parser.add_argument(
        '--num-hidden-layers',
        type=int,
        default=12,
        help="Num hidden layers to use for AST (max 12)"
    )

    parser.add_argument(
        '--disable-progress-bar',
        default=False,
        type=bool,
        help='Disable tqdm progress bar while training',
        choices=[
            'True',
            'False',
        ]
    )

    parser.add_argument(
        '--exclude-cities',
        default="None",
        type=str,
        help='Exclude a City from training',
        choices=[
            "barcelona",
            "helsinki",
            "lisbon",
            "london",
            "lyon",
            "milan",
            "paris",
            "prague",
            "stockholm",
            "vienna",
            "None",
        ]
    )

    args = parser.parse_args()
    run_training(args)
