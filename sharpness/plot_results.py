import os, sys, csv
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import matplotlib.ticker as mticker
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
import seaborn as sns
from sklearn.linear_model import LinearRegression

from scipy.stats import spearmanr
# enable command line argument parsing
from argparse import ArgumentParser

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# def custom_multiple_subplots(path, nfigs_height, nfigs_width, Xs, Ys, labels, title=[], font_size=12, x_axis="", y_axis="", y_lims=[], symbols = ["o-", "x-", "v-"], fig_size=(10,4), colors=[], line_styles=[]):
#     fig, fig_subplots = plt.subplots(nfigs_height, nfigs_width, figsize=fig_size)
    
#     for plot_nr in range(len(Xs)):
#         Xs_line = Xs[plot_nr]
#         Ys_line = Ys[plot_nr]
#         for line_nr in range(len(Xs_line)):
#             fig_subplots[plot_nr].plot(Xs_line[line_nr], Ys_line[line_nr], symbols[line_nr], label=labels[line_nr])
#             fig_subplots[plot_nr].set(xlabel=x_axis, ylabel=y_axis)
#             #fig_subplots.set_xscale("log")
#             fig_subplots[plot_nr].set_xscale("symlog")
#             if len(y_lims) > 0:
#                 fig_subplots[plot_nr].set_ylim((0.725, 0.925))
#             #fig_subplots.xaxis.set_minor_formatter(mticker.ScalarFormatter())
#             fig_subplots[plot_nr].xaxis.set_major_formatter(mticker.ScalarFormatter())
#             fig_subplots[plot_nr].xaxis.get_major_formatter().set_scientific(False)
#             fig_subplots[plot_nr].xaxis.get_major_formatter().set_useOffset(False)
#             # font = {'family': 'serif',
#             #     'color':  'black',
#             #     'weight': 'normal',
#             #     'size': 18,
#             # }
#             fig_subplots[plot_nr].set_xticks(Xs_line[line_nr])
#             fig_subplots[plot_nr].set_title(titles[plot_nr])#,fontdict=font)
#             fig_subplots[plot_nr].legend(loc="lower right")
#     plt.tight_layout()
#     plt.savefig(path)





# def make_nice_line_plot(path ,Xs, Ys, labels, title="", font_size=12, x_axis="", y_axis="", fig_size=[], colors=[], line_styles=[]):
#     #plt.figure(figsize=fig_size)
#     plt.rcParams.update({'font.size': font_size})
#     plt.title(title)
#     plt.xlabel(x_axis, fontsize=font_size)
#     plt.ylabel(y_axis, fontsize=font_size)
#     for i in range(len(Xs)):
#         X = Xs[i]
#         Y = Ys[i]
#         label = labels[i]
#         if len(colors) == 0:
#             plt.plot(X, Y, label=label)
#         else:
#             plt.plot(X, Y, color=colors[i], linestyle=line_styles[i], label=label)


#     plt.legend()
#     plt.savefig(path)
#     plt.clf()

def get_df_rq(df, n_rq):
    row_start = df.index[df['Model'] == "RQ" + str(n_rq) + " Start"].tolist()[0]
    row_end = df.index[df['Model'] == "RQ" + str(n_rq) + " End"].tolist()[0]
    df_rq = df.iloc[row_start + 1:row_end]
    #df_rq = df_rq[df_rq[] > 0]
    df_rq['Test (all)'] = df_rq['Test (all)'].astype(float)
    df_rq = df_rq[df_rq['Test (all)'] >= 0.5]
    print("Shape for RQ" + str(n_rq) + ": " + str(df_rq.shape))
    #print(df_rq)
    return df_rq

def scatter_plot(dest_path, df, separator, x_column, y_column):
    
    # print(x_column)
    for sep_value in sep_values:
        df_part = df[df[separator] == sep_value]
        x = df_part[x_column].values
        #print(x_values)
        y = df_part[y_column].values.astype(np.float32)
        #print(y_values)
        plt.scatter(x, y)
        #for i in range(len(x)):
            #print(x[i],y[i])
    #plt.xscale("symlog")
    plt.ylim((0.5,0.6))
    plt.savefig(dest_path)
    plt.clf()
    #plt.clf()

def plot_histogram_with_error(dest_path, df, mean_column, std_column):    
    fs = 11
    mean_values = df[mean_column].values.astype(np.float32)
    std_values = df[std_column].values.astype(np.float32)
    sorted_idx = np.argsort(mean_values)
    mean_sorted = mean_values[sorted_idx]
    std_sorted = std_values[sorted_idx]
    fig, ax = plt.subplots()
    x_pos = np.arange(len(mean_values))
    ax.bar(x_pos, mean_sorted,
       yerr=std_sorted,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=5)
    ax.set_title('Sharpness Distribution', fontsize=fs+2)
    ax.set_ylabel('Mean Sharpness', fontsize=fs)
    ax.set_xlabel('Model ID', fontsize=fs)
    ax.set_xticks(x_pos, fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_xticklabels(x_pos, rotation=90)

    
    plt.tight_layout()

    plt.savefig(dest_path)
    plt.show()
    #sns.barplot()

def create_bar_plot_disaggregated(plot_path, data, labels, group_names, log_scale=False, colors=[], ylim=None, title="", ylabel=""):
    """
    Creates a bar plot with one group of bars per array of values in `data`
    and corresponding labels in `labels`. Each bar in each group has height
    equal to the value in the numpy array and has the label on the bottom.
    """
    # Determine the number of groups and bars per group
    num_groups = len(data)
    data_max = 0.
    data_min = 10.
    for i in range(num_groups):
        new_max = np.max(data[i])
        data_max = np.max((data_max, new_max))
        new_min = np.min(data[i])
        data_min = np.min((data_min, new_min))
    fs = 16
    # Set the bar width
    bar_width = 0.25

    # Create the figure and axes objects
    fig, ax = plt.subplots(figsize=(8, 5))

    tick_labels = []
    num_ticks = 0
    x_ticks = []
    shift = 0
    # Loop over the groups of bars and plot them
    for i in range(num_groups):
        # Create a list of x-coordinates for the bars
        num_bars = len(data[i])
        x = np.arange(num_bars)

        # Shift the x-coordinates for each group of bars
        
        x_shifted = (x + shift) * bar_width
        x_ticks += x_shifted.tolist()

        # Get the values and labels for this group of bars
        values = data[i]
        bar_labels = labels[i]

        # Plot the bars for this group
        if len(colors) > 0:
            ax.bar(x_shifted, values, bar_width, color = colors[i])
        else:
            ax.bar(x_shifted, values, bar_width)

        tick_labels += labels[i].tolist()
        # Set the labels for each bar
        group_x = x_shifted[0] + (x_shifted[-1] - x_shifted[0])/2
        if ylim != None:
            ax.text(group_x, ylim[1] + 0.07 * (ylim[1] - ylim[0]), group_names[i], ha='center', va='bottom', fontsize=fs)
        else:
            ax.text(group_x, data_max + 0.1 * (data_max - data_min), group_names[i], ha='center', va='bottom', fontsize=fs)
        # for j, value in enumerate(values):
            # ax.text(group_x, value, group_names, ha='center', va='bottom')
        
        shift += (num_bars + 1) 

    # Set the x-ticks and labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=fs)
    #ax.set_title(title)
    ax.set_ylabel(ylabel, fontsize=fs)
    if ylim != None:
        ax.set_ylim(ylim)
    if log_scale:
        plt.yscale("symlog")
    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.clf()


# def create_bar_plot(plot_path, data, labels):
#     """
#     Creates a bar plot with one group of bars per array of values in `data`
#     and corresponding labels in `labels`. Each bar in each group has height
#     equal to the value in the numpy array and has the label on the bottom.
#     """
#     # Determine the number of groups and bars per group
#     num_groups = len(data)
    
#     # Set the bar width
#     bar_width = 0.35

    
    

#     # Create the figure and axes objects
#     fig, ax = plt.subplots()

#     # Loop over the groups of bars and plot them
#     for i in range(num_groups):
#         # Create a list of x-coordinates for the bars
#         num_bars = len(data[i])
#         x = np.arange(num_bars)

#         # Shift the x-coordinates for each group of bars
#         shift = i * bar_width
#         x_shifted = x + shift

#         # Get the values and labels for this group of bars
#         values = data[i]
#         bar_labels = labels[i]

#         # Plot the bars for this group
#         ax.bar(x_shifted, values, bar_width, label=bar_labels)

#     # Set the x-ticks and labels
#     ax.set_xticks(x + bar_width * (num_groups - 1) / 2)
#     #ax.set_xticklabels(labels[0])

#     ax.legend()
#     plt.savefig(plot_path)
#     plt.clf()


def plot_histogram_criterion(dest_path, df, mean_column, separators, log_scale=False, colors=[], ylim=None, title="", ylabel=""):
    label_lists = []
    value_lists = []
    for separator in separators:
        sep_values = df[separator].unique()
        labels = []
        values = []
        for sep_value in sep_values:
            df_part = df[df[separator] == sep_value]
            x_values = df_part[mean_column].values.astype(np.float32)
            
            if sep_value != 'gdtuo Adam':
                if separator == 'Learning Rate':
                    labels.append(str(to_exp_notation(float(sep_value))))
                else:
                    labels.append(sep_value)
            else:
                labels.append('GDTUO')
            values.append(np.mean(x_values))
        value_lists.append(np.array(values))
        label_lists.append(np.array(labels))
    separators_adjusted = [x if x != 'Optimizer' else 'Optimiser' for x in separators]
    create_bar_plot_disaggregated(dest_path, value_lists, label_lists, separators_adjusted, colors=colors, ylim=ylim, log_scale=log_scale, title=title, ylabel=ylabel)

def plot_results_regression(dest_path, df, separator, y_column, criteriums = ['Mean Sharpness', 'Max Sharpness']):
    if separator != "":
        sep_values = df[separator].unique()
    # # print(x_column)
    # for sep_value in sep_values:
        
    #     x = df_part[x_column].values
    #     #print(x_values)
    #     y = df_part[y_column].values.astype(np.float32)
    devices = [
    ('S4', 'mediumblue'),
    ('S5', 'green'),
    ('S6', 'orange')
    ]
    fs = 20

    
    # criteriums = ['Max Sharpness', 'Mean Sharpness']
    fig, axes = plt.subplots(len(criteriums), 1, figsize=(10, 5 * len(criteriums)), sharey=True)
    for index, criterium in enumerate(criteriums):
        ax = axes[index]
        if separator != "":
            for sep_value in sep_values:
                df_part = df[df[separator] == sep_value]
                x_values = df_part[criterium].values.astype(np.float32)
                y_values = df_part[y_column].values
                y_values = y_values.astype(np.float32)
                plot = sns.regplot(
                    #data=df,
                    #x=criterium,
                    x= x_values,
                    y= y_values,
                    ci=None,
                    #y=y_column,
                    ax=ax,
                    label=sep_value
                )
                plot.set(ylim =(0.52,0.63))
        else:
            df_part = df
            x_values = df_part[criterium].values.astype(np.float32)
            y_values = df_part[y_column].values
            y_values = y_values.astype(np.float32)
            plot = sns.regplot(
                #data=df,
                #x=criterium,
                x= x_values,
                y= y_values,
                #y=y_column,
                ax=ax
            )
            plot.set(ylim =(0.51,0.63))
        sns.despine(ax=ax)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.tick_params(axis='both', which='minor', labelsize=fs)
        if index == 0:
            ax.set_ylabel('Accuracy', fontsize=fs)
        else:
            ax.set_ylabel('Accuracy', fontsize=fs)
            ax.tick_params(
                axis='y',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False
            ) 
            if separator != "":
                ax.legend(
                    title=separator, 
                    loc='center left', 
                    bbox_to_anchor=(1, 0.5), 
                    fontsize=fs
                )
        ax.set_xlabel(criterium, fontsize=fs)
    plt.tight_layout()
    plt.savefig(dest_path)
    plt.clf()

def to_exp_notation(num):
    if num == 0:
        return "0"
    exponent = 0
    while abs(num) < 1:
        num *= 10
        exponent -= 1
    while abs(num) >= 10:
        num /= 10
        exponent += 1
    return "1e{}".format(exponent)

def calc_stuff(x, y):
    # Computes the metrics CCC, PCC, and RMSE between the sequences x and y
    #  CCC:  Concordance correlation coeffient
    #  PCC:  Pearson's correlation coeffient
    #  RMSE: Root mean squared error
    # Input:  x,y: numpy arrays (one-dimensional)
    # Output: CCC,PCC,RMSE

    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)

    covariance = np.nanmean((x - x_mean) * (y - y_mean))

    x_var = np.nanmean((x_mean - x) ** 2)
    y_var = np.nanmean((y_mean - y) ** 2)

    CCC = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)

    x_std = np.sqrt(x_var)
    y_std = np.sqrt(y_var)

    PCC = covariance / (x_std * y_std)

    RMSE = np.sqrt(np.nanmean((x - y) ** 2))

    scores = np.array([CCC, PCC, RMSE])

    return scores

def calc_scores(path, df):
    mean_sharpness_values = df['Mean Sharpness'].values.astype(np.float32)
    max_sharpness_values = df['Max Sharpness'].values.astype(np.float32)
    test_error_values = df['Test (all)'].values.astype(np.float32)
    mean_pcc = pearsonr(mean_sharpness_values, test_error_values)
    max_pcc = pearsonr(max_sharpness_values, test_error_values)
    # _, mean_pcc, _ = calc_stuff(mean_sharpness_values, test_error_values)
    # _, max_pcc, _ = calc_stuff(max_sharpness_values, test_error_values)
    print("Mean PCC:" + str(mean_pcc[0]))
    print("Max PCC:" + str(max_pcc[0]))



if __name__ == "__main__":
    code_path = os.path.dirname(os.path.realpath(__file__))
    
    # command line arg parsing
    
    
    parser = ArgumentParser()
    #my_root_dir = os.path.join(os.path.dirname(code_path),"Aufnahmen_bearbeitet/20190420_Easter_special/Images_Part_2_DcAMP")
    parser.add_argument('--result_dir', default="result_plots/")
    parser.add_argument('--result_file', default="Interspeech Curvature Results - Sheet1.csv")
    args = parser.parse_args()
    

    os.makedirs(args.result_dir, exist_ok=True)
    df_org = pd.read_csv(args.result_file)
    #x = df.columns
    #print(x)
    df_rqs = []
    dfs_rqs = []
    for n_rq in range(1, 4):
        df_rq = get_df_rq(df_org, n_rq)
        if len(df_rqs) == 0:
            df_rqs = df_rq
        else:
            df_rqs = pd.concat((df_rqs, df_rq))
        dfs_rqs.append(df_rq)

    for n_rq in [1, 5]:
        rq_dir = args.result_dir + "RQ" + str(n_rq) + "/"
        if n_rq == 1:
            df = df_rqs
        else:
            df = get_df_rq(df_org, 5)
        color_separators = ["Model", "Batch Size", "Learning Rate", "Optimizer"]
        colors = ["springgreen", "wheat", "skyblue", "salmon"] 
        
        # x_columns = ["Mean Sharpness"]
        
        #result_dir = rq_dir + x_column + "/"
        os.makedirs(rq_dir, exist_ok=True)
        # seaborn_dir = rq_dir + "regplot/"
        # os.makedirs(seaborn_dir, exist_ok=True)
        for color_separator in color_separators:
            plot_path = rq_dir + color_separator + "_RQ" + str(n_rq) + ".pdf"
            #scatter_plot(plot_path, df_rq, color_separator, x_column, "Test (all)")
            plot_results_regression(plot_path, df, color_separator, "Test (all)")
        plot_path_single = rq_dir + "RQ" + str(n_rq) + ".pdf"
        plot_results_regression(plot_path_single, df, "", "Test (all)")
        plot_path_single_OOD = rq_dir + "RQ" + str(n_rq) + "_OOD.pdf"
        plot_results_regression(plot_path_single_OOD, df, "", "Test (all)")
        plot_path_histogram_error = rq_dir + "histogram_error.pdf"
        plot_histogram_with_error(plot_path_histogram_error, df, 'Mean Sharpness', 'Std Sharpness')
        plot_path_histogram_criterion = rq_dir + "histogram_criterion.pdf"
        plot_histogram_criterion(plot_path_histogram_criterion, df, 'Mean Sharpness', color_separators, colors = colors, log_scale=False,title="Disaggregated Mean Sharpness", ylabel="Average Mean Sharpness")
        plot_path_histogram_disacc = rq_dir + "histogram_disaggregated_accuracies.pdf"
        plot_histogram_criterion(plot_path_histogram_disacc, df, 'Test (all)', color_separators, colors=colors, ylim=(0.52, 0.62), title="Disaggregated Accuracy", ylabel="Average Accuracy")
        score_path = rq_dir + "scores.csv"
        calc_scores(score_path, df)

# RQ 5

                
        
