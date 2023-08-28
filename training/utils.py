import audmetric
import audobject
#from calculate_different_sharpness_values import calculate_sharpness
import torch
import tqdm
import pandas as pd
from PIL import Image
from torchinfo import summary
import numpy as np


def transfer_features(features, device):
    return features.to(device).float()

def get_output_dim(model):
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Linear):
            # Found a linear layer
            output_dim = module.out_features
            break
    return output_dim

def get_df_from_dataset(dataset):
    data = []
    for i in range(len(dataset)):
        image, label = dataset[i]
        # image = image.numpy().flatten()  # Flatten the image tensor
        data.append((image, label))
    data_dict = {'image': [], 'label': []}
    for image, label in data:
        data_dict['image'].append(image)
        data_dict['label'].append(label)
    df = pd.DataFrame(data_dict)
    return df


def evaluate_categorical(model, device, loader, transfer_func, disable, criterion):
    metrics = {
        'UAR': audmetric.unweighted_average_recall,
        'ACC': audmetric.accuracy,
        'F1': audmetric.unweighted_average_fscore
    }

    model.to(device)
    model.eval()
    #print("Getting the output dims")
    output_dim = get_output_dim(model)
    #print(output_dim)
    # outputs = torch.zeros((len(loader.dataset), model.output_dim))
    outputs = torch.zeros((len(loader.dataset), output_dim))
    targets = torch.zeros(len(loader.dataset))
    with torch.no_grad():
        for index, (features, target) in tqdm.tqdm(
            enumerate(loader),
            desc='Batch',
            total=len(loader),
            disable=disable,
        ):
            start_index = index * loader.batch_size
            end_index = (index + 1) * loader.batch_size
            if end_index > len(loader.dataset):
                end_index = len(loader.dataset)
            outputs[start_index:end_index, :] = model(
                transfer_func(features, device))
            targets[start_index:end_index] = target
    loss = criterion(outputs, targets.type(torch.LongTensor))
    targets = targets.numpy()
    outputs = outputs.cpu()
    predictions = outputs.argmax(dim=1).numpy()
    outputs = outputs.numpy()
    return {
        key: metrics[key](targets, predictions)
        for key in metrics.keys()
    }, targets, predictions, outputs, loss.item()


class LabelEncoder(audobject.Object):
    def __init__(self, labels):
        self.labels = sorted(labels)
        codes = range(len(labels))
        self.inverse_map = {code: label for code,
                            label in zip(codes, labels)}
        self.map = {label: code for code,
                    label in zip(codes, labels)}

    def encode(self, x):
        return self.map[x]

    def decode(self, x):
        return self.inverse_map[x]


def disaggregated_evaluation(df, groundtruth, task, stratify, evaluation_type: str = 'regression'):
    if evaluation_type == 'regression':
        metrics = {
            'CC': audmetric.pearson_cc,
            'CCC': audmetric.concordance_cc,
            'MSE': audmetric.mean_squared_error,
            'MAE': audmetric.mean_absolute_error
        }
    elif evaluation_type == 'categorical':
        metrics = {
            'UAR': audmetric.unweighted_average_recall,
            'ACC': audmetric.accuracy,
            'F1': audmetric.unweighted_average_fscore
        }
    else:
        raise NotImplementedError(evaluation_type)

    df = df.reindex(groundtruth.index)
    results = {key: {} for key in metrics.keys()}
    for key in metrics.keys():
        # print(df)
        # print("groundtruth")
        # print(groundtruth.shape)
        # print(groundtruth)
        results[key]['all'] = metrics[key](
            groundtruth[task],
            df['predictions']
        )
        for stratifier in stratify:
            for variable in groundtruth[stratifier].unique():
                indices = groundtruth.loc[groundtruth[stratifier]
                                          == variable].index
                results[key][variable] = metrics[key](
                    groundtruth.reindex(indices)[task],
                    df.reindex(indices)['predictions']
                )

    return results

    

class GrayscaleToRGB(object):
    def __call__(self, image):
        # Convert tensor to numpy array and adjust data type if needed
        # print(image.shape)
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        image = image.astype(np.float32)

        # Check if the input is already a grayscale image
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)

        # Convert grayscale image to RGB
        image_rgb = np.repeat(image, 3, axis=0)
        # print(image_rgb.shape)
        # Convert the RGB image to a tensor
        # image_tensor = torch.from_numpy(image_rgb)
        # print(image_tensor.shape)
        return image_rgb

#########################################################################################################
### Util Functions from https://github.com/tml-epfl/sharpness-vs-generalization                       ###
#########################################################################################################


import torch
import torch.nn.functional as F
from datetime import datetime


def process_arg(args, arg):
    if arg in ['gpu', 'eval_sharpness', 'log', 'rewrite']:
        return ''
    if arg == 'adaptive':
        return ''
    if arg != 'model_path':
        return str(getattr(args, arg))
    # return args.model_path.split('/')[-1][:24].replace(' ', '_')
    return ''


def get_path(args, log_folder):
    name = '-'.join([process_arg(args, arg) for arg in list(filter(lambda x: x not in ['adaptive'], vars(args)))])
    name = str(datetime.now())[:-3].replace(' ', '_') + name
    if getattr(args, 'adaptive'):
        name += '-adaptive'
    path = f'{log_folder}/{name}.json'
    return path


def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
        

def compute_err(batches, model, loss_f=F.cross_entropy, n_batches=-1):
    n_wrong_classified, train_loss_sum, n_ex = 0, 0.0, 0

    with torch.no_grad():
        for i, (X, _, y, _, ln) in enumerate(batches):
            if n_batches != -1 and i > n_batches:  # limit to only n_batches
                break
            X, y = X.cuda(), y.cuda()
            
            # print(X, X.shape)
            output = model(X)
            loss = loss_f(output, y)  

            n_wrong_classified += (output.max(1)[1] != y).sum().item()
            train_loss_sum += loss.item() * y.size(0)
            n_ex += y.size(0)

    err = n_wrong_classified / n_ex
    avg_loss = train_loss_sum / n_ex

    return err, avg_loss


def estimate_loss_err(model, batches, loss_f):
    err = 0
    loss = 0
    
    with torch.no_grad():
        for i_batch, (x, _, y, _, _) in enumerate(batches):
            x, y = x.cuda(), y.cuda()
            curr_y = model(x)
            loss += loss_f(curr_y, y)
            err += (curr_y.max(1)[1] != y).float().mean().item()
            
    return loss.item() / len(batches), err / len(batches)

def batches_from_dataloader(loader, test=False):
    batches = []
    for index, (features, targets) in tqdm.tqdm(
                enumerate(loader),
                desc=f'Epoch {0}',
                total=len(loader)
        ):
        batches.append((features, targets))
        if test and index > 2:
            break
    return batches 
