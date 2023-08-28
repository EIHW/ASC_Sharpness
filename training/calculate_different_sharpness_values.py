import torch
import models
from neuralbench import GridSearchModule, OptimizerWrapper, ExcludeSearch, GDTUOWrapper, SAMWrapper
from utils import get_output_dim, evaluate_categorical, transfer_features, batches_from_dataloader
import tqdm
import pandas as pd
from datasets import CachedDataset
import sharpness_adaptive
from torchinfo import summary
import os
from torch.optim import Adam, SGD
from utils import LabelEncoder

# import argparse
# import copy
# import h5py
# import torch
# import time
# import socket
# import os
# import sys
# import numpy as np
# import torchvision
# import torch.nn as nn
# import dataloader
# import evaluation
# import projection as proj
# import net_plotter
# import plot_2D
# import plot_1D
# import model_loader
# import scheduler

# from DCASE2020.datasets import CachedDataset, LabelEncoder
# import pandas as pd
# import plot_surface_folder

# TODO: Change back adaptive to False?
def grad_norm(model, device, adaptive = False):
    #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
    for p in model.parameters():
        x = p.grad
    norm = torch.norm(
                torch.stack([
                    ((torch.abs(p) if adaptive else 1.0) * p.grad).norm(p=2).to(device)
                    #((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for p in model.parameters()
                ]),
                p=2
            )
    return norm

# def get_sharpness_keskar(model, data, epsilon=1e-2):
#     """
#     Github code from https://github.com/darylchang123/ml-experiments/blob/master/utils/ml_utils.py 

#     This function computes the sharpness of a minimizer by maximizing the loss in a neighborhood around the minimizer.
#     Based on sharpness metric defined in https://arxiv.org/pdf/1609.04836.pdf.
    
#     :param model: model, where the weights represent a minimizer of the loss function
#     :param data: data to evaluate the model on
#     :param epsilon: controls the size of the neighborhood to explore
#     :return: sharpness
#     """
#     # Get original loss
#     original_loss, original_accuracy = model.evaluate(data)
    
#     # Compute bounds on weights
#     weights = model.get_weights()
#     weight_shapes = [w.shape for w in weights]
#     flattened_weights = np.concatenate([x.flatten()for x in weights])
#     delta = epsilon * (np.abs(flattened_weights) + 1)
#     lower_bounds = flattened_weights - delta 
#     upper_bounds = flattened_weights + delta
    
#     # Create copy of model so we don't modify original
#     model.save('pickled_objects/sharpness_model_clone.h5')
#     model_clone = keras.models.load_model('pickled_objects/sharpness_model_clone.h5')
#     os.remove('pickled_objects/sharpness_model_clone.h5')
    
#     # Minimize
#     x, f, d = scipy.optimize.fmin_l_bfgs_b(
#         func=get_negative_loss,
#         fprime=get_negative_loss_gradient,
#         x0=flattened_weights,
#         args=(model_clone, data, weight_shapes),
#         bounds=list(zip(lower_bounds, upper_bounds)),
#         maxiter=10,
#         maxls=1,
#         disp=1,
#     )
    
#     # Compute sharpness
#     sharpness = (-f - original_loss) / (1 + original_loss) * 100
#     return sharpness


#todo  
def calculate_sharpness(model, device, loader, transfer_func, disable, criterion, sharp_measures=["filter-normalized-epsilon", "adaptive", "taylor"], rho_list =[0.002, 0.05]):
    sharpness_values = {}
    if "adaptive" in sharp_measures:
        print("adaptive")
        results, _, predictions, outputs, train_loss = evaluate_categorical(
                model,
                device,
                loader,
                transfer_features,
                True,
                criterion
            )
        # Put test=False for the full evaluation
        batches = batches_from_dataloader(loader, test=True)
        # sharpness_obj, sharpness_err, _, output = sharpness_adaptive.eval_APGD_sharpness(
        #     model, batches, criterion, 1 - results["UAR"], train_loss, n_iters=20, return_output=True, rho=0.002
        #     # rho=args.rho, n_iters=args.n_iters, n_restarts=args.n_restarts, step_size_mult=args.step_size_mult,
        #     # rand_init=args.sharpness_rand_init, no_grad_norm=args.no_grad_norm,
        #     # verbose=True, return_output=True, adaptive=args.adaptive, version='default', norm='l2'
        #     )
        rho = rho_list[0]
        print("calculate adaptive sharpness (modern look paper)")
        sharpness_obj, sharpness_err, _, output = sharpness_adaptive.eval_APGD_sharpness(
            model, batches, criterion, 0., 0., n_iters=20, return_output=True, rho=rho
            # rho=args.rho, n_iters=args.n_iters, n_restarts=args.n_restarts, step_size_mult=args.step_size_mult,
            # rand_init=args.sharpness_rand_init, no_grad_norm=args.no_grad_norm,
            # verbose=True, return_output=True, adaptive=args.adaptive, version='default', norm='l2'
            )

        # TODO: Figure out what to use here
        #sharpness_values["adaptive"] = sharpness_obj#, sharpness_err
        sharpness_values["adaptive"] = sharpness_err
    else:
        sharpness_values["adaptive"] = 0
        # so far only taylor is implemented according to SAM idea
    # if "taylor" in sharp_measures:
    #     print("taylor")
    #     model.to(device)
    #     model.eval()
    #     output_dim = get_output_dim(model)
    #     outputs_minimum = torch.zeros((len(loader.dataset), output_dim))
    #     targets_minimum = torch.zeros(len(loader.dataset))
    #     outputs_taylor = torch.zeros((len(loader.dataset), output_dim))
    #     targets_taylor = torch.zeros(len(loader.dataset))

    #     rho = rho_list[1]
    #     #with torch.no_grad():
    #     assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
    #     #defaults = dict(rho=rho, adaptive=adaptive)
    #     # loop over dataset
    #     for index, (features, target) in tqdm.tqdm(
    #         enumerate(loader),
    #         desc='Batch',
    #         total=len(loader),
    #         disable=disable,
    #     ):
    #         start_index = index * loader.batch_size
    #         end_index = (index + 1) * loader.batch_size
    #         if end_index > len(loader.dataset):
    #             end_index = len(loader.dataset)
    #         # move uphill with taylor series (dual norm solution): w --> w + eps
    #         original_params = [param.clone() for param in model.parameters()]
    #         gnorm = grad_norm(model, device)
    #         scale = rho / (gnorm + 1e-12)
    #         for param in model.parameters():
    #             # Apply the same change as in line 24 of your original code
    #             param.data = param.grad * scale.to(param) + param

    #             # for group in model.param_groups:
    #             #     scale = group["rho"] / (grad_norm + 1e-12)

    #             #     for p in group["params"]:
    #             #         if p.grad is None: continue
    #             #         self.state[p]["old_p"] = p.data.clone()
    #             #         e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
    #             #         p.add_(e_w)  # climb to the local maximum "w + e(w)"
    #             outputs_taylor[start_index:end_index, :] = model(transfer_func(features, device))
    #             targets_taylor[start_index:end_index] = target


    #             # move back to original weights: w + eps --> w
    #             for param, original_param in zip(model.parameters(), original_params):
    #                 param.data.copy_(original_param)
    #             outputs_minimum[start_index:end_index, :] = model(transfer_func(features, device))
    #             targets_minimum[start_index:end_index] = target
    #             # break

    #         # loss values and difference thereof for sharpness
    #         loss_taylor = criterion(outputs_taylor, targets_taylor.type(torch.LongTensor)).item()
    #         loss_minimum = criterion(outputs_minimum, targets_minimum.type(torch.LongTensor)).item()
    #         taylor_sharpness = loss_taylor - loss_minimum
    #         sharpness_values["taylor"] = taylor_sharpness
    # else:
    #     sharpness_values["taylor"] = 0
            # TODO: Fix the batching! Maybe have batch-size all data, or average the gradient!
    if "taylor" in sharp_measures:
        print("taylor")
        rho = rho_list[1]
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        sam_optim = SAMWrapper(SGD, momentum=0.9, rho=rho)
        optimizer = sam_optim.create(
                model, lr=rho, device=device)
        model.to(device)
        model.eval()
        output_dim = get_output_dim(model)
        outputs_minimum = torch.zeros((len(loader.dataset), output_dim))
        targets_minimum = torch.zeros(len(loader.dataset))
        outputs_taylor = torch.zeros((len(loader.dataset), output_dim))
        targets_taylor = torch.zeros(len(loader.dataset))
        
        
        #with torch.no_grad():
        
        #defaults = dict(rho=rho, adaptive=adaptive)
        # loop over dataset
        for index, (features, targets) in tqdm.tqdm(
            enumerate(loader),
            desc='Batch',
            total=len(loader),
            disable=disable,
        ):
            start_index = index * loader.batch_size
            end_index = (index + 1) * loader.batch_size
            if end_index > len(loader.dataset):
                end_index = len(loader.dataset)
            # move uphill with taylor series (dual norm solution): w --> w + eps
            original_params = [param.clone() for param in model.parameters()]
            output = model(transfer_features(features, device))
            targets = targets.to(device)
            outputs_minimum[start_index:end_index, :] = output.detach()
            targets_minimum[start_index:end_index] = targets
            
            # first forward-backward pass
            loss = criterion(output, targets)  # use this loss for any training statistics
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            outputs_taylor[start_index:end_index, :] = model(transfer_func(features, device)).detach()
            targets_taylor[start_index:end_index] = targets
            
            


            # move back to original weights: w + eps --> w
            for param, original_param in zip(model.parameters(), original_params):
                param.data.copy_(original_param)
            # outputs_minimum[start_index:end_index, :] = model(transfer_func(features, device))
            # targets_minimum[start_index:end_index] = target
            # break

            # loss values and difference thereof for sharpness
        loss_taylor = criterion(outputs_taylor, targets_taylor.type(torch.LongTensor)).item()
        loss_minimum = criterion(outputs_minimum, targets_minimum.type(torch.LongTensor)).item()
        taylor_sharpness = loss_taylor - loss_minimum
        sharpness_values["taylor"] = taylor_sharpness
    else:
        sharpness_values["taylor"] = 0
    
    return sharpness_values

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

if __name__ == '__main__':
    # Test
    model_path = "/nas/staff/data_work/manuel/cloned_repos/visualisation/results/test/run06/cnn10_None_Adam_0-001_32_100_42_None_None/state.pth.tar"
    model_name = "cnn10"
    dataset = "dcase"
    batch_size = 32

    device = "cpu"


    # load dataset:
    df_train = pd.read_csv(
    os.path.join(
            "/data/eihw-gpu5/milliman/DCASE/DCASE2020/metadata",
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
            "/data/eihw-gpu5/milliman/DCASE/DCASE2020/metadata",
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
            "/data/eihw-gpu5/milliman/DCASE/DCASE2020/metadata",
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

    n_classes = len(df_train['scene_label'].unique())
    encoder = LabelEncoder(
        list(df_train['scene_label'].unique()))

    features = pd.read_csv("/data/eihw-gpu5/milliman/DCASE/DCASE2020/mel_spectrograms/features.csv").set_index('filename')

    # * custom feature path support
    # if args.custom_feature_path is not None:
    #     features = replace_file_path(
    #         features, "features", args.custom_feature_path)

    db_args = {
        'features': features,
        'target_column': 'scene_label',
        'target_transform': encoder.encode,
        'feature_dir': "/data/eihw-gpu5/milliman/DCASE/DCASE2020/mel_spectrograms/"
    }

    db_class = CachedDataset
    criterion = torch.nn.CrossEntropyLoss()
    train_dataset = db_class(
            df_train,
            **db_args
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4
    )
    
    
    # load model
    model = models.load(dataset, model_name, model_path)
    summary(model=model, 
        input_size=((1,1,1001,64)), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )

    # batches_sharpness = []
    # loss_f = []
    # train_err = []
    # train_loss = []

    results, _, predictions, outputs, train_loss = evaluate_categorical(
                model,
                device,
                train_loader,
                transfer_features,
                True,
                criterion
            )
    batches = batches_from_dataloader(train_loader, test=False)
    sharpness_obj, sharpness_err, _, output = sharpness_adaptive.eval_APGD_sharpness(
        model, batches, criterion, 1 - results["UAR"], train_loss, n_iters=20, return_output=True, rho=0.002
        # rho=args.rho, n_iters=args.n_iters, n_restarts=args.n_restarts, step_size_mult=args.step_size_mult,
        # rand_init=args.sharpness_rand_init, no_grad_norm=args.no_grad_norm,
        # verbose=True, return_output=True, adaptive=args.adaptive, version='default', norm='l2'
        )
    print(sharpness_err)
    print(sharpness_obj)