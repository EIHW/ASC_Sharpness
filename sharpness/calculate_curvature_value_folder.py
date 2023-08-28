"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""
import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
import dataloader
import evaluation
import projection as proj
import net_plotter
import plot_2D
import plot_1D
import model_loader
import scheduler
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.ticker as mticker
import mpi4pytorch as mpi
from os.path import splitext, basename
import pathlib

# change back for DCASE2022
#from DCASE2022.datasets import CacdhedDataset, LabelEncoder
from DCASE2020.datasets import CachedDataset, LabelEncoder
import pandas as pd

def custom_multiple_subplots(path, nfigs_height, nfigs_width, Xs, Ys, labels, titles, font_size=12, x_axis="", y_axis="", y_lims=[], symbols = ["o-", "x-", "v-"], fig_size=(10,4), colors=[], line_styles=[], scale_log=False, detailed_ticks=True):
    fig, fig_subplots = plt.subplots(nfigs_height, nfigs_width, figsize=fig_size)
    
    for plot_nr in range(len(Xs)):
        Xs_line = Xs[plot_nr]
        Ys_line = Ys[plot_nr]
        for line_nr in range(len(Xs_line)):
            x = Xs_line[line_nr]
            y = Ys_line[line_nr]
            symbol = symbols[line_nr]
            label = labels[line_nr]
            fig_subplots[plot_nr].plot(x, y, symbol, label=label)
            fig_subplots[plot_nr].set(xlabel=x_axis, ylabel=y_axis)
            #fig_subplots.set_xscale("log")
            if scale_log == True:
                fig_subplots[plot_nr].set_xscale("symlog")
            if len(y_lims) > 0:
                fig_subplots[plot_nr].set_ylim((0.725, 0.925))
            #fig_subplots.xaxis.set_minor_formatter(mticker.ScalarFormatter())
        
            fig_subplots[plot_nr].xaxis.set_major_formatter(mticker.ScalarFormatter())
            fig_subplots[plot_nr].xaxis.get_major_formatter().set_scientific(False)
            fig_subplots[plot_nr].xaxis.get_major_formatter().set_useOffset(False)
            # font = {'family': 'serif',
            #     'color':  'black',
            #     'weight': 'normal',
            #     'size': 18,
            # }
            if detailed_ticks == True:
                fig_subplots[plot_nr].set_xticks(Xs_line[line_nr])
            fig_subplots[plot_nr].set_title(titles[plot_nr])#,fontdict=font)
            fig_subplots[plot_nr].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)


def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c

def calculate_sharpness(xdata, ydata, func, plot_file, ylim=-1, eps=0.25, second_dim=False):
    x_for_plot = []
    y_for_plot = []
    # sharpness analysis
    if not second_dim:
        y_min = np.min(ydata)
        y_min_idx = np.argmin(ydata)
        x_stepsize = (xdata[-1] - xdata[0]) / len(xdata)
        eps_steps = int(eps / x_stepsize)
        start_idx = y_min_idx - eps_steps
        minimum_offset = xdata[y_min_idx]

        print("Minimum offset: " + str(minimum_offset))
        if start_idx < 0:
            start_idx = 0
        end_idx = y_min_idx + eps_steps
        y_eps_interval = ydata[start_idx:end_idx]
        y_max_eps = np.max(y_eps_interval)
        sharpness = (y_max_eps - y_min) / (1 + y_min) * 100
        print("Sharpness: " + str(sharpness))
        y_data_eps = np.ones(xdata.shape) * y_max_eps
        x_for_plot.append([xdata, xdata])
        y_for_plot.append([ydata, y_data_eps])

        # Parabola Analysis
        y_data_for_fit = np.copy(ydata)
        xdata_for_fit = np.copy(xdata)
        if ylim > -1:
            reduced_x = [] 
            reduced_y = []
            for i in range(len(xdata_for_fit)):
                if y_data_for_fit[i] <= ylim + y_min:
                    reduced_x.append(xdata_for_fit[i])
                    reduced_y.append(y_data_for_fit[i])
            xdata_for_fit = np.array(reduced_x)
            y_data_for_fit = np.array(reduced_y)
        popt, pcov = curve_fit(func, xdata_for_fit, y_data_for_fit)
        curvature = popt[0]
        y_fitted = func(xdata_for_fit, popt[0], popt[1], popt[2])
        print("Parameters of the Parabola Fit" + str(popt))
        x_for_plot.appen([xdata_for_fit, xdata_for_fit])
        y_for_plot.append([y_data_for_fit, y_fitted])

        custom_multiple_subplots(plot_file, 2, 1, x_for_plot, y_for_plot, ["Data", "Fitted"], ["Parabola Fit", "Eps-Sharpness"], fig_size=(4,8), symbols=["-", "-"], detailed_ticks=False)
        return popt[0], sharpness, minimum_offset

    else:
        distances = xdata.reshape(1, len(xdata))**2 + xdata.reshape(len(xdata),1)**2
        mask = distances <= eps**2
        y_masked = np.where(mask, ydata, np.min(ydata))
        y_max = np.max(y_masked)
        y_min = np.min(y_masked)
        sharpness = (y_max - y_min) / (1 + y_min) * 100
        print("Sharpness: " + str(sharpness))
        return 0, sharpness, 0

    #print(popt)
    

def name_surface_file(args, dir_file):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    # dataloder parameters
    if args.raw_data: # without data normalization
        surf_file += '_rawdata'
    if args.data_split > 1:
        surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)
    surf_file += "_" + args.partition

    return surf_file + ".h5"


def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)
            return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=int(args.xnum))
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=int(args.ynum))
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file

def custom_sort2(file_paths, importance, ascending):
    """
    Sorts a list of file paths based on the name of the last folder, splitting it by '_'
    and using custom importance and sort order for each part of the string.

    :param file_paths: List of file paths to sort
    :param importance: List of integers indicating the importance of each part of the folder name
                       (the first integer is the most important, the last integer is the least important)
    :param ascending: List of booleans indicating whether each part of the folder name should be sorted
                      in ascending order (True) or descending order (False)
    :return: A sorted list of file paths
    """
    # Define a function to extract the relevant folder name and split it into parts
    def key_function(path):
        folder_name = path.split('/')[-2]  # Get the second-to-last element of the path (the last folder name)
        return [part.strip() for part in folder_name.split('_')]  # Split the folder name by '_' and remove whitespace

    # Define a function to compare two folder names based on their importance and sort order
    def compare_parts(part1, part2):
        imp1 = importance[len(part1)-1]  # Get the importance of the current part
        imp2 = importance[len(part2)-1]  # Get the importance of the other part
        if imp1 > imp2:
            return 1
        elif imp1 < imp2:
            return -1
        elif ascending[len(part1)-1]:
            return part1 < part2
        else:
            return part1 > part2

    # Sort the list of file paths using the key and comparison functions
    return sorted(file_paths, key=key_function, cmp=compare_parts)


def custom_sort(strings, order):
    
    for i in range(len(strings)):

        strings[i] = basename(strings[i]).split('_')
    strings.sort(key=lambda x: [order.index(part) for part in x])
    for i in range(len(strings)):
        strings[i] = '_'.join(strings[i])
    return strings

def compare_surf_files(file1, file2, order, ascending):
    
    folder1_to_sort = file1.split('/')[-2]
    folder2_to_sort = file2.split('/')[-2]
    
    file1_parts = folder1_to_sort.split('_')
    file2_parts = folder2_to_sort.split('_')

    for i in range(len(order)):
        order_by = order[i]
        if file1_parts[order_by] < file2_parts[order_by]:
            return not ascending[i]
        elif file1_parts[order_by] > file2_parts[order_by]:
            return ascending[i]
    return False

def sort_surf_files(surf_files, order, ascending):
    # ugly workaround, bubblesort XD
    sorted = False
    while not sorted:
        sorted = True
        for i in range(len(surf_files) - 1):
            swap = compare_surf_files(surf_files[i], surf_files[i + 1], order, ascending)
            if swap:
                temp = surf_files[i]
                surf_files[i] = surf_files[i + 1]
                surf_files[i + 1] = temp
                sorted = False
    return surf_files
            
    


def crunch(surf_file, loss_key, acc_key, comm, rank, args):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """

    f = h5py.File(surf_file, 'r')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    # if loss_key not in f.keys():
    #     shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
    #     losses = -np.ones(shape=shape)
    #     accuracies = -np.ones(shape=shape)
    #     if rank == 0:
    #         f[loss_key] = losses
    #         f[acc_key] = accuracies
    # else:

    losses = f[loss_key][:]
    accuracies = f[acc_key][:]
    plot_file = splitext(surf_file)[0] + "_parabola.pdf"
    curvature, sharpness, minimum_offset = calculate_sharpness(xcoordinates, losses, parabola, plot_file, ylim=5, second_dim=args.second_dim)
    #fitted_curvature = pops[0]


    results_file = splitext(surf_file)[0] + "_curvature_result.csv"
    with open(results_file, "w") as f:
        f.write("Curvature,Sharpness,Miimum Offset\n")
        f.write(str(curvature) + "," + str(sharpness) + "," + str(minimum_offset))
    f.close()
    return curvature, sharpness, minimum_offset


#--------------------------------------------------------------------------
# Setup dataloader
#--------------------------------------------------------------------------
# download CIFAR10 if it does not exit



###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    
    # data parameters
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--datapath', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--data-root', default='/data/eihw-gpu5/trianand/DCASE/d22-t1/TAU-urban-acoustic-scenes-2022-mobile-development', help='path to the data on local device')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--disaggregated',  default=False, action='store_true',)
    parser.add_argument('--features', default='/data/eihw-gpu5/trianand/DCASE/d22-t1/TAU-urban-acoustic-scenes-2022-mobile-development', help='path to the features on local device')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')
    
    

    # model parameters
    parser.add_argument('--model', default='resnet56', help='model name')
    parser.add_argument('--model_folder', default='', help='the common folder that contains all the models')
    parser.add_argument('--model_filename', default='state.pth.tar', help='default model file name')
    parser.add_argument('--model_file', default='', help='path to the trained model file')
    parser.add_argument('--model_file2', default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model_file3', default='', help='use (model_file3 - model_file) as the ydirection')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')
    parser.add_argument('--partition', default='train', help='lon which partition should it be plotted')

    # direction parameters
    parser.add_argument('--dir_file', default='', help='specify the name of direction file, or the path to an eisting direction file')
    parser.add_argument('--dir_type', default='weights', help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default=None, help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False, help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')
    parser.add_argument('--second_dim', action='store_true', default=False, help='if it is 2D, the metric gets calculated differently')

    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')

    args = parser.parse_args()

    surf_files = []
    for root, dirs, files in os.walk(args.model_folder):
        for file in files:
            if file.endswith(args.partition + ".h5"):
                surf_files.append(os.path.abspath(os.path.join(root, file)))
    
    #surf_files.sort()
    surf_files = sort_surf_files(surf_files, [0, 2, 4, 6, 3], [True, True, True, True, False])
    #surf_files = custom_sort2(surf_files, [0, 2, 4, 3, 6], [True, True, True, False, True])

    results_file_all = args.model_folder + "curvature_result.csv"
    with open(results_file_all, "w") as f:
        f.write("Model,Curvature,Sharpness,Minimum Offset\n")
    
    for surf_file in surf_files:
        #--------------------------------------------------------------------------
        # Environment setup
        #--------------------------------------------------------------------------
        print("-" * 60)
        print(surf_file)
        if args.mpi:
            comm = mpi.setup_MPI()
            rank, nproc = comm.Get_rank(), comm.Get_size()
        else:
            comm, rank, nproc = None, 0, 1


        #--------------------------------------------------------------------------
        # Check plotting resolution
        #--------------------------------------------------------------------------
        try:
            args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
            args.ymin, args.ymax, args.ynum = (None, None, None)
            if args.y:
                args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
                assert args.ymin and args.ymax and args.ynum, \
                'You specified some arguments for the y axis, but not all'
        except:
            raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')


        

        #--------------------------------------------------------------------------
        # Start the computation
        #--------------------------------------------------------------------------
        
        if not args.disaggregated:
            curvature, sharpness, minimum_offset = crunch(surf_file, 'train_loss', 'train_acc', comm, rank, args)
            with open(results_file_all, "a") as f:
                f.write(pathlib.PurePath(surf_file).parent.name + "," + str(curvature) + "," + str(sharpness) + "," + str(minimum_offset) + "\n")
            
            # crunch(surf_file, net, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, args)

            #--------------------------------------------------------------------------
            # Plot figures
            #--------------------------------------------------------------------------
            if args.plot and rank == 0:
                if args.y and args.proj_file:
                    plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, 'train_loss', args.show)
                elif args.y:
                    plot_2D.plot_2d_contour(args, surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
                else:
                    plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)
        
        else:
            for i, rec_device in enumerate(rec_devices):
                print(rec_device)
                crunch(surf_files[i], net, w, s, ds[i], dataloaders[i], 'train_loss', 'train_acc', comm, rank, args)
            # crunch(surf_file, net, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, args)

            #--------------------------------------------------------------------------
            # Plot figures
            #--------------------------------------------------------------------------
                if args.plot and rank == 0:
                    if args.y and args.proj_file:
                        plot_2D.plot_contour_trajectory(surf_files[i], dir_file, args.proj_file, 'train_loss', args.show)
                    elif args.y:
                        plot_2D.plot_2d_contour(args, surf_files[i], 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
                    else:
                        plot_1D.plot_1d_loss_err(surf_files[i], args.xmin, args.xmax, args.loss_max, args.log, args.show)