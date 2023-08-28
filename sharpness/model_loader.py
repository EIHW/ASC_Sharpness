import os
import torch
import cifar10.model_loader
from DCASE2022.models.models import Cnn10, Cnn14

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    elif dataset == 'dcase':
        #out_dim for dcase is 10
        out_dim = 10
        if model_name == "cnn10":
            net = Cnn10(out_dim)
        elif model_name == "cnn14":
            net = Cnn14(out_dim)
        net.load_state_dict(torch.load(model_file))
        net.eval()
    return net
