# Evaluation of Minimum Sharpness

This repository adapts and extends the repository for the visualisation of landscapes from [*Visualizing the Loss Landscape of Neural Nets*](https://arxiv.org/abs/1712.09913) to calculate the an epsilon-sharpness in a 1D or 2D loss landscape. Pretrained models are available for the DCASE2020-Task1 acoustic scene classification. The remainder of this README contains relevant parts of the original Readme and additions resulting from made changes.

# Visualizing the Loss Landscape of Neural Nets

This repository contains the PyTorch code for the paper
> Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. [*Visualizing the Loss Landscape of Neural Nets*](https://arxiv.org/abs/1712.09913). NIPS, 2018.

An [interactive 3D visualizer](http://www.telesens.co/loss-landscape-viz/viewer.html) for loss surfaces has been provided by [telesens](http://www.telesens.co/2019/01/16/neural-network-loss-visualization/).

Given a network architecture and its pre-trained parameters, this tool calculates and visualizes the loss surface along random direction(s) near the optimal parameters.
The calculation can be done in parallel with multiple GPUs per node, and multiple nodes.
The random direction(s) and loss surface values are stored in HDF5 (`.h5`) files after they are produced.

## Setup

**Environment**: One or more multi-GPU node(s) with the following software/libraries installed:
- [PyTorch 0.4](https://pytorch.org/)
- [openmpi 3.1.2](https://www.open-mpi.org/)
- [mpi4py 2.0.0](https://mpi4py.scipy.org/docs/usrman/install.html)
- [numpy 1.15.1](https://docs.scipy.org/doc/numpy/user/quickstart.html)  
- [h5py 2.7.0](http://docs.h5py.org/en/stable/build.html#install)
- [matplotlib 2.0.2](https://matplotlib.org/users/installing.html)
- [scipy 0.19](https://www.scipy.org/install.html)

**Pre-trained models**:
The code accepts pre-trained PyTorch models for the DCASE2020 (as well as the original CIFAR10) dataset.
To load the pre-trained model correctly, the model file should contain `state_dict`, which is saved from the `state_dict()` method.
Some pre-trained models (CNN10 and CNN14 PANNs) for the DCASE2020 models can be downloaded here.
- [DCASE2020 models]()

## Visualizing 1D or 2D loss curve for several models

You can plot the 1D or 2D visualisations for several models, each having their own subfolder, collected in the same folder by running
```
mpirun -n 1 python plot_surface_folder.py --cuda --partition train --dataset dcase --x=-0.25:0.25:11 --y=-0.25:0.25:11 --dir_type weights --data-root /path/to/metadata/ --features /path/to/extracted_mel_spectrograms/features.csv --model_folder /path/to/pretrained_models/ --xnorm filter --xignore biasbn --ngpu 1 --plot --random_seed=44 --loss_max 5 --ynorm filter --yignore biasbn
```

## Calculate the sharpness values for a folder of models
```
python calculate_curvature_value_folder --cuda --mpi --partition train --dataset dcase --x=-0.25:0.25:3 --dir_type states --data-root /path/to/metadata/ --features /path/to/features/features.csv --model_folder /path/to/pretrained_models/ --xnorm filter --xignore biasbn --ngpu 1 --second_dim --batch_size 16 --n_seeds 10 random_seed 42 --no_random_seed
```

## Citation

If you find this code useful in your research, please also cite the original paper:

```
@inproceedings{visualloss,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Neural Information Processing Systems},
  year={2018}
}
```
