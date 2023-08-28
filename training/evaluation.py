from sincnet import (
    SincNet,
    MLP
)
from models import (
    Cnn10,
    Cnn14,
    create_ResNet50_model,
    ModifiedEfficientNet
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
from calculate_different_sharpness_values import calculate_sharpness
from gradient_descent_the_ultimate_optimizer.gdtuo import ModuleWrapper, NoOpOptimizer
from torchinfo import summary

import warnings
# Ignore Using backward() UserWarning of gdtuo
warnings.filterwarnings(category=UserWarning, action="ignore")

def get_experiment_info(experiment_folder):
    # experimentfolder: str containing experiment details, e.g. some_root/CIFAR10_cnn10_pretrained-False_None_Adam_0-001_32_100_42_None_None/
    exp_subdir = os.path.basename(experiment_folder)
    exp_det = exp_subdir.split("_")
    dataset = exp_det[0]
    approach = exp_det[1]
    pretrained = exp_det[2] == "pretrained-True"
    category = exp_det[3]
    if category == "None":
        category = None
    optimizer = exp_det[4]
    lr = float(exp_det[5].replace("-", "."))
    bs = int(exp_det[6])
    epochs = int(exp_det[7])
    seed = int(exp_det[8])
    lr_scheduler = exp_det[9]
    exclude_cities = exp_det[10]
    last_model_path = experiment_folder + "Epoch_" + str(epochs) + "/state.pth.tar"
    best_model_path =  experiment_folder + "/state.pth.tar"
    return dataset, approach, pretrained, category, optimizer, lr, bs, epochs, seed, lr_scheduler, exclude_cities

# def evaluate_model(model, state_file, loader, device, criterion, sharp_measures=["taylor"]):
# def evaluate_model(model, state_file, loader, device, criterion, sharp_measures=["adaptive"]):
def evaluate_model(model, state_file, loader, device, criterion, sharp_measures=["adaptive", "taylor"]):
    print("Evaluation")
    checkpoint = torch.load(state_file, map_location=torch.device(device))
    state_dict = checkpoint#['model']
    model.load_state_dict(state_dict, strict=True)
    print("model loaded")
    results, targets, predictions, outputs, loss = evaluate_categorical(
        model, device, loader, transfer_features, True, criterion)
    print("results")
    print(results)
    print("loss")
    print(loss)
    sharpness_values = calculate_sharpness(model, device, loader, transfer_features, True, criterion, sharp_measures=sharp_measures, rho_list=[0.002,0.005])
    print("Sharpness")
    print(sharpness_values)
    print("Done")
    return results, loss, sharpness_values[sharp_measures[0]], sharpness_values[sharp_measures[1]] 
    # return results, loss, sharpness_values[sharp_measures[0]], 0 
    


def run_evaluation(args):
    
    def _get_device_multiprocessing(device):
        torch.cuda.set_device(torch.cuda.device(device))
        return "cuda:"+str(torch.cuda.current_device())
    args.device = args.device if isinstance(
        args.device, str) else _get_device_multiprocessing(args.device)

    experiments_path = args.results_root
    experiment_folders = [f.path for f in os.scandir(experiments_path) if f.is_dir()]
    # print(experiment_folders)
        

    evaluation_folder = experiments_path + "combined_evaluation/"
    evaluation_table_outpath = evaluation_folder + "overall_evaluation.csv"
    os.makedirs(evaluation_folder, exist_ok=True)
    evaluation_df = pd.DataFrame()
    evaluation_df['folder_name'] = []
    evaluation_df['dataset'] = []
    evaluation_df['approach'] = []
    evaluation_df['pretrained'] = []
    evaluation_df['optimizer'] = []
    evaluation_df['lr'] = []
    evaluation_df['bs'] = []
    evaluation_df['epochs'] = []
    evaluation_df['seed'] = []
    evaluation_df['train_loss_last'] = []
    evaluation_df['dev_loss_last'] = []
    evaluation_df['test_loss_last'] = []
    evaluation_df['train_result_last'] = []
    evaluation_df['dev_result_last'] = []
    evaluation_df['test_result_last'] = []
    evaluation_df['train_sharpness_adaptive_last'] = []
    evaluation_df['dev_sharpness_adaptive_last'] = []
    evaluation_df['test_sharpness_adaptive_last'] = []
    evaluation_df['train_sharpness_taylor_last'] = []
    evaluation_df['dev_sharpness_taylor_last'] = []
    evaluation_df['test_sharpness_taylor_last'] = []
    evaluation_df['train_loss_best'] = []
    evaluation_df['dev_loss_best'] = []
    evaluation_df['test_loss_best'] = []
    evaluation_df['train_result_best'] = []
    evaluation_df['dev_result_best'] = []
    evaluation_df['test_result_best'] = []
    evaluation_df['train_sharpness_adaptive_best'] = []
    evaluation_df['dev_sharpness_adaptive_best'] = []
    evaluation_df['test_sharpness_adaptive_best'] = []
    evaluation_df['train_sharpness_taylor_best'] = []
    evaluation_df['dev_sharpness_taylor_best'] = []
    evaluation_df['test_sharpness_taylor_best'] = []
    #dataset, approach, pretrained, category, optimizer, lr, bs, epochs, seed, lr_scheduler, exclude_cities

    previous_dataset = ""
    for experiment_folder in experiment_folders:
        if os.path.basename(experiment_folder) == "combined_evaluation":
            continue
        print("-"*50)
        print("-"*50)
        print("Experiment: " + experiment_folder)
        dataset, approach, pretrained, category, optimizer, learning_rate, batch_size, epochs, seed, lr_scheduler, exclude_cities = get_experiment_info(experiment_folder)
        torch.manual_seed(seed)
        gen_seed = torch.Generator().manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
        ### DCASE
        if dataset == 'DCASE2020':
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
            # only load dataset if it wasn't already loaded for the previous loading step
            # TODO: For some evaluations maybe we need to adjust this? 
            if dataset != previous_dataset:
                df_train = pd.read_csv(
                    os.path.join(
                        args.data_root,
                        'evaluation_setup',
                        'fold1_train.csv'
                    ), sep='\t').set_index('filename')
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
                    ), sep='\t').set_index('filename')
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
                    ), sep='\t').set_index('filename')
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

                if category is not None:
                    df_train = df_train.loc[df_train['scene_category'] == category]
                    df_dev = df_dev.loc[df_dev['scene_category'] == category]
                    df_test = df_test.loc[df_test['scene_category'] == category]

                if exclude_cities != "None":
                    df_train = df_train.loc[~df_train["city"].isin(exclude_cities)]

                n_classes = len(df_train['scene_label'].unique())
                encoder = LabelEncoder(
                    list(df_train['scene_label'].unique()))

                features = pd.read_csv(args.features).set_index('filename')


                db_args = {
                    'features': features,
                    'target_column': 'scene_label',
                    'target_transform': encoder.encode,
                    'feature_dir': args.feature_dir
                }
            
            if approach == 'cnn14':
                model = Cnn14(
                    output_dim=n_classes
                )
                db_class = CachedDataset
                model.to_yaml(os.path.join(experiment_folder, 'model.yaml'))
                # criterion = torch.nn.CrossEntropyLoss()
            
            # Load model
            elif approach == 'cnn10':
                model = Cnn10(
                    output_dim=n_classes
                )
                db_class = CachedDataset
                model.to_yaml(os.path.join(experiment_folder, 'model.yaml'))
                # criterion = torch.nn.CrossEntropyLoss()
            elif approach.startswith("efficientnet"):
                model = ModifiedEfficientNet(n_classes, scaling_type=approach)
                db_class = CachedDataset
                db_args['transform'] = transforms.Compose([GrayscaleToRGB()])
                # criterion = torch.nn.CrossEntropyLoss()
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
        elif dataset == "CIFAR10":
            # Parameters for development set
            devel_percentage = 0.2
            if approach in ["cnn10", "cnn14"]:
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
            if dataset != previous_dataset:
                train_dev_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform)
                # it is very important to keep the manual seed here the same as in the training
                generator1 = torch.Generator().manual_seed(42)
                train_dataset, dev_dataset = torch.utils.data.random_split(train_dev_dataset, [1 - devel_percentage, devel_percentage], generator=generator1)
            
            
                test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

                df_dev = get_df_from_dataset(dev_dataset)
                df_test = get_df_from_dataset(test_dataset)
                
                encoder = LabelEncoder(
                    list(test_dataset.class_to_idx.keys()))
                n_classes = len(test_dataset.class_to_idx.keys())
                input_channels = 3

            if approach == 'cnn14':
                model = Cnn14(
                    output_dim=n_classes,
                    in_channels=input_channels
                )
                model.to_yaml(os.path.join(experiment_folder, 'model.yaml'))
            elif approach == 'cnn10':
                model = Cnn10(
                    output_dim=n_classes,
                    in_channels=input_channels
                )
                model.to_yaml(os.path.join(experiment_folder, 'model.yaml'))
            elif approach == 'ResNet50':
                model = create_ResNet50_model(n_classes)
            elif approach.startswith("efficientnet"):
                model = ModifiedEfficientNet(n_classes, scaling_type=approach)

        
        
        
                    
        # Print a summary using torchinfo (uncomment for actual output)
        criterion = torch.nn.CrossEntropyLoss()        
        
        x, y = train_dataset[0]
        x = np.expand_dims(x, axis=0)
        print(x.shape)
        # summary(model=model, 
        #     input_size=(x.shape), # make sure this is "input_size", not "input_shape"
        #     # col_names=["input_size"], # uncomment for smaller output
        #     col_names=["input_size", "output_size", "num_params", "trainable"],
        #     col_width=20,
        #     row_settings=["var_names"]
        # )
        # print("-" * 50)
        # personalized_plot_model(model)

        if approach == 'sincnet':
            db_args.pop('transform')
        
        # create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            # shuffle = False because of evaluation
            shuffle=False,
            batch_size=batch_size,
            num_workers=4,
            generator=gen_seed
        )

        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            shuffle=False,
            batch_size=1 if approach == 'sincnet' else batch_size,
            num_workers=4,
            generator=gen_seed
        )
        

        # df_dev = pd.DataFrame(dev_dataset.dataset)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=1 if approach == 'sincnet' else batch_size,
            num_workers=4,
            generator=gen_seed
        )

        sharp_measures = ["adaptive", "taylor"]
        # last model state (after epoch 100)
        print("-"*50)
        print("Evaluate Last model")
        state_file_last = experiment_folder + "/state.pth.tar"
        print("Train")
        train_results_last, train_loss_last, train_sharpness_adaptive_last, train_sharpness_taylor_last =  evaluate_model(model, state_file_last, train_loader, args.device, criterion)
        print("Dev")
        dev_results_last, dev_loss_last, dev_sharpness_adaptive_last, dev_sharpness_taylor_last =  evaluate_model(model, state_file_last, dev_loader, args.device, criterion)
        print("Test")
        test_results_last, test_loss_last, test_sharpness_adaptive_last, test_sharpness_taylor_last =  evaluate_model(model, state_file_last, test_loader, args.device, criterion)
        
        # best epoch state
        print("Evaluate Best Model")
        state_file_best = experiment_folder + "/Epoch_" + str(epochs) + "/state.pth.tar"
        print("Train")
        train_results_best, train_loss_best, train_sharpness_adaptive_best, train_sharpness_taylor_best =  evaluate_model(model, state_file_best, train_loader, args.device, criterion)
        print("Dev")
        dev_results_best, dev_loss_best, dev_sharpness_adaptive_best, dev_sharpness_taylor_best =  evaluate_model(model, state_file_best, dev_loader, args.device, criterion)
        print("Test")
        test_results_best, test_loss_best, test_sharpness_adaptive_best, test_sharpness_taylor_best =  evaluate_model(model, state_file_best, test_loader, args.device, criterion)
        evaluation_df = evaluation_df.append(
        {'folder_name': experiment_folder,
        'dataset': dataset,
        'approach': approach,
        'pretrained': pretrained,
        'optimizer': optimizer,
        'lr': learning_rate,
        'bs': batch_size,
        'epochs': epochs,
        'seed': seed,
        'train_loss_last': train_loss_last,
        'dev_loss_last': dev_loss_last,
        'test_loss_last': test_loss_last,
        'train_result_last': train_results_last,
        'dev_result_last': dev_results_last,
        'test_result_last': test_results_last,
        'train_sharpness_adaptive_last': train_sharpness_adaptive_last,
        'dev_sharpness_adaptive_last': dev_sharpness_adaptive_last,
        'test_sharpness_adaptive_last': test_sharpness_adaptive_last,
        'train_sharpness_taylor_last': train_sharpness_taylor_last,
        'dev_sharpness_taylor_last': dev_sharpness_taylor_last,
        'test_sharpness_taylor_last': test_sharpness_taylor_last,
        'train_loss_best': train_loss_best,
        'dev_loss_best': dev_loss_best,
        'test_loss_best': test_loss_best,
        'train_result_best': train_results_best,
        'dev_result_best': dev_results_best,
        'test_result_best': test_results_best,
        'train_sharpness_adaptive_best': train_sharpness_adaptive_best,
        'dev_sharpness_adaptive_best': dev_sharpness_adaptive_best,
        'test_sharpness_adaptive_best': test_sharpness_adaptive_best,
        'train_sharpness_taylor_best': train_sharpness_taylor_best,
        'dev_sharpness_taylor_best': dev_sharpness_taylor_best,
        'test_sharpness_taylor_best': test_sharpness_taylor_best
        }, ignore_index=True)
        evaluation_df.to_csv(evaluation_table_outpath, index=False)

    print("Evaluation Done!")

        # accuracy_history = []
        # uar_history = []
        # f1_history = []
        # train_loss_history = []
        # valid_loss_history = []

        # if not os.path.exists(os.path.join(experiment_folder, 'state.pth.tar')):
            
        #     print("Training was not done in ", experiment_folder)
        #     # TODO: Here! 
        #     encoder.to_yaml(os.path.join(experiment_folder, 'encoder.yaml'))
        #     with open(os.path.join(experiment_folder, 'hparams.yaml'), 'w') as fp:
        #         yaml.dump(vars(args), fp)

        #     writer = SummaryWriter(log_dir=os.path.join(experiment_folder, 'log'))

        #     torch.save(
        #         model.state_dict(),
        #         os.path.join(
        #             experiment_folder,
        #             'initial.pth.tar')
        #     )
        #     if isinstance(args.optimizer, str):
        #         optim = args.optimizer
        #         if args.optimizer == 'SGD':
        #             optimizer = torch.optim.SGD(
        #                 model.parameters(),
        #                 momentum=0.9,
        #                 lr=args.learning_rate
        #             )
        #         elif args.optimizer == 'Adam':
        #             optimizer = torch.optim.Adam(
        #                 model.parameters(),
        #                 lr=args.learning_rate
        #             )
        #         elif args.optimizer == 'RMSprop':
        #             optimizer = torch.optim.RMSprop(
        #                 model.parameters(),
        #                 lr=args.learning_rate,
        #                 alpha=.95,
        #                 eps=1e-7
        #             )
        #     else:
        #         optimizer = args.optimizer.create(
        #             model, lr=args.learning_rate, device=args.device)

        #     if not "sheduler_wrapper" in args or args.sheduler_wrapper == None:
        #         sheduler = None
        #     elif isinstance(args.sheduler_wrapper, list):
        #         sheduler = []
        #         for sh in args.sheduler_wrapper:
        #             sheduler.append(sh.create(optimizer))
        #     else:
        #         sheduler = args.sheduler_wrapper.create(optimizer)

            

        #     max_metric = -1
        #     best_epoch = 0
        #     best_state = None
        #     best_results = None

        #     # accuracy_history = []
        #     # uar_history = []
        #     # f1_history = []
        #     # train_loss_history = []
        #     # valid_loss_history = []

        #     for epoch in range(epochs):
        #         model.to(device)
        #         model.train()
        #         epoch_folder = os.path.join(
        #             experiment_folder,
        #             f'Epoch_{epoch+1}'
        #         )
        #         os.makedirs(epoch_folder, exist_ok=True)

        #         if "train_timer" in args:
        #             args.train_timer.start()
        #         _loss_history = []
        #         for index, (features, targets) in tqdm.tqdm(
        #             enumerate(train_loader),
        #             desc=f'Epoch {epoch}',
        #             total=len(train_loader),
        #             disable=args.disable_progress_bar
        #         ):

        #             if (features != features).sum():
        #                 raise ValueError(features)

        #             if isinstance(optimizer, ModuleWrapper):
        #                 loss = train_step_gdtuo(
        #                     model, optimizer, criterion, features, targets, device)
        #             elif isinstance(optimizer, (KFACOptimizer, EKFACOptimizer)):
        #                 loss = train_step_kfac(
        #                     model, optimizer, criterion, features, targets, device, epoch+1, index+1)
        #             elif isinstance(optimizer, SAM):
        #                 loss = train_step_SAM(
        #                     model, optimizer, criterion, features, targets, device)
        #             else:
        #                 loss = train_step_normal(
        #                     model, optimizer, criterion, features, targets, device)
        #             if index % 50 == 0:
        #                 writer.add_scalar(
        #                     'Loss',
        #                     loss,
        #                     global_step=epoch * len(train_loader) + index
        #                 )
        #             _loss_history.append(loss)
                    
        #         train_loss = sum(_loss_history)/len(_loss_history)
        #         # print(train_loss)
        #         if "train_timer" in args:
        #             args.train_timer.stop()

        #         if "valid_timer" in args:
        #             args.valid_timer.start()
        #         ##  Sharpness

        #         # sharpness_values = calculate_sharpness(model, device, train_loader, transfer_features, args.disable_progress_bar, criterion)
        #         # print(sharpness_values)
                



        #         # dev set evaluation
        #         results, _, predictions, outputs, valid_loss = evaluate_categorical(
        #             model,
        #             device,
        #             dev_loader,
        #             transfer_features,
        #             args.disable_progress_bar,
        #             criterion
        #         )
        #         results_df = pd.DataFrame(
        #             index=df_dev.index,
        #             data=predictions,
        #             columns=['predictions']
        #         )
        #         results_df['predictions'] = results_df['predictions'].apply(
        #             encoder.decode)
        #         results_df.reset_index().to_csv(os.path.join(epoch_folder, 'dev.csv'), index=False)
        #         np.save(os.path.join(epoch_folder, 'outputs.npy'), outputs)
        #         if args.dataset == "DCASE2020":
        #             # print(results_df)
        #             logging_results = disaggregated_evaluation(
        #                 results_df,
        #                 df_dev,
        #                 'scene_label',
        #                 ['scene_category', 'city', 'device'],
        #                 'categorical'
        #             )

        #             with open(os.path.join(epoch_folder, 'dev.yaml'), 'w') as fp:
        #                 yaml.dump(logging_results, fp)
        #             for metric in logging_results.keys():
        #                 writer.add_scalars(
        #                     f'dev/{metric}',
        #                     logging_results[metric],
        #                     (epoch + 1) * len(train_loader)
        #                 )

        #         torch.save(model.cpu().state_dict(), os.path.join(
        #             epoch_folder, 'state.pth.tar'))
        #         results["train_loss"] = train_loss
        #         results["val_loss"] = valid_loss
                
        #         print(f'Dev results at epoch {epoch+1}:\n{yaml.dump(results)}')
        #         # save accuracy metric
        #         accuracy_history.append(results["ACC"])
        #         uar_history.append(results["UAR"])
        #         f1_history.append(results["F1"])
        #         train_loss_history.append(train_loss)
        #         valid_loss_history.append(valid_loss)
        #         if results['ACC'] > max_metric:
        #             max_metric = results['ACC']
        #             best_epoch = epoch
        #             best_state = model.cpu().state_dict()
        #             best_results = results.copy()

        #         # plateau_scheduler.step(results['ACC'])
        #         if "valid_timer" in args:
        #             args.valid_timer.stop()
        #     train_results, _, _, _, train_loss = evaluate_categorical(
        #             model,
        #             device,
        #             train_loader,
        #             transfer_features,
        #             args.disable_progress_bar,
        #             criterion
        #         )
        #     print(f'Final Train results:\n {yaml.dump(train_results)}')
        #     print(f'Final Train loss:\n {yaml.dump(train_loss)}')
        #     #results["sharpness_value"] = sharpness_values
        #     print("Final Train results: ", train_results)
        #     print("Final Train loss: ", train_loss)

        #     sharpness_values = calculate_sharpness(model, device, train_loader, transfer_features, args.disable_progress_bar, criterion)
        #     print(f'Sharpness:\n{yaml.dump(results)}')
        #     #results["sharpness_value"] = sharpness_values
        #     print("Sharpness Value: ", sharpness_values)
        #     print(
        #         f'Best dev results found at epoch {best_epoch+1}:\n{yaml.dump(best_results)}')
        #     best_results['Epoch'] = best_epoch + 1
        #     with open(os.path.join(experiment_folder, 'dev.yaml'), 'w') as fp:
        #         yaml.dump(best_results, fp)
        #     writer.close()
        # else:
        #     best_state = torch.load(os.path.join(
        #         experiment_folder, 'state.pth.tar'))
        #     print('Training already run')
        #     epoch_folder = os.path.join(
        #             experiment_folder,
        #             f'Epoch_{epochs}'
        #         )
        # if dataset == "DCASE2020": 
        #     print("saving to: ", os.path.join(experiment_folder, 'test_holistic.yaml'))
        #     if not os.path.exists(os.path.join(experiment_folder, 'test_holistic.yaml')):
        #         model.load_state_dict(best_state)
        #         test_results, targets, predictions, outputs, valid_loss = evaluate_categorical(
        #             model, device, test_loader, transfer_features, args.disable_progress_bar, criterion)
        #         print(f'Best test results:\n{yaml.dump(test_results)}')
        #         torch.save(best_state, os.path.join(
        #             experiment_folder, 'state.pth.tar'))
        #         np.save(os.path.join(experiment_folder, 'targets.npy'), targets)
        #         np.save(os.path.join(experiment_folder, 'outputs.npy'), outputs)
        #         np.save(os.path.join(experiment_folder, 'predictions.npy'), outputs)
        #         results_df = pd.DataFrame(
        #             index=df_test.index,
        #             data=predictions,
        #             columns=['predictions']
        #         )
        #         results_df['predictions'] = results_df['predictions'].apply(
        #             encoder.decode)
        #         results_df.reset_index().to_csv(os.path.join(epoch_folder, 'test.csv'), index=False)
        #         with open(os.path.join(experiment_folder, 'test.yaml'), 'w') as fp:
        #             yaml.dump(test_results, fp)
        #             logging_results = disaggregated_evaluation(
        #                 results_df,
        #                 df_test,
        #                 'scene_label',
        #                 ['scene_category', 'city', 'device'],
        #                 'categorical'
        #             )
        #             with open(os.path.join(experiment_folder, 'test_holistic.yaml'), 'w') as fp:
        #                 yaml.dump(logging_results, fp)
        #     else:
        #         print('Evaluation already run')
        #         # in case we don't have any training the benchrunner doesn't make a lot of sense.


        # previous_dataset = dataset
        # return accuracy_history, uar_history, f1_history, train_loss_history, valid_loss_history



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
        '--feature_dir',
        default=''
    )
    
    parser.add_argument(
        '--device',
        help='CUDA-enabled device to use for training',
        required=True
    )
    
    args = parser.parse_args()
    run_evaluation(args)
