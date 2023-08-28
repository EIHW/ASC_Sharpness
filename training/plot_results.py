import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import argparse
import json
import warnings
from scipy.stats import pearsonr, spearmanr
# Ignore Using backward() UserWarning of gdtuo
warnings.filterwarnings(category=UserWarning, action="ignore")

def plot_results(args, correlation_file, out_filename, column_x='train_sharpness_adaptive_last', column_y='test_loss_last', y_dict="", threshold_column='train_loss_last', threshold_x = 0.03):
    evaluation_folder = args.results_root + "combined_evaluation/"

    evaluation_file = evaluation_folder + "overall_evaluation.csv"
    

    df = pd.read_csv(evaluation_file)
    print("Before ", df.shape)
    df_low_loss = df[df[threshold_column] < threshold_x]
    df_high_loss = df[df[threshold_column] > threshold_x]
    print("After low: ", df_low_loss.shape)
    print("After high: ", df_high_loss.shape)
    #df = pd.read_csv('your_file.csv')

# Set up the figure and axis
    fig, ax = plt.subplots()
    
    # Plot the data with logarithmic scales
    # ax.set_yscale('log')  # Set logarithmic scale for the y-axis
    # ax.set_xscale('log')  # Set logarithmic scale for the x-axis
    x = df_low_loss[column_x].values
    if y_dict == "":
        y = df_low_loss[column_y]
          # Replace 'column1' and 'column2' with your actual column names
    else:
        y = []
        for y_string in df_low_loss[column_y].values:
            y_dictionary = json.loads(y_string.replace("'", "\""))
            y.append(y_dictionary[y_dict])
    ax.plot(x, y, 'o')
    # Customize the plot
    ax.set_xlabel(column_x)
    ax.set_ylabel(column_y)
    ax.set_title('Logarithmic Plot')

    plt.savefig(evaluation_folder + out_filename + '_low-loss.png')

    pearson_corr, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)
    print(column_x + column_y + "_low-loss," + str(pearson_corr) + "," + str(spearman_corr))
    with open(correlation_file, "a") as f:
        f.writelines(column_x + column_y + "_low-loss," + str(pearson_corr) + "," + str(spearman_corr) + "\n")
    


    x = df_high_loss[column_x].values
    if y_dict == "":
        y = df_high_loss[column_y].values
    else:
        y = []
        for y_string in df_high_loss[column_y].values:
            y_dictionary = json.loads(y_string.replace("'", "\""))
            y.append(y_dictionary[y_dict])
    ax.plot(x, y, 'o')


    # Customize the plot
    ax.set_xlabel(column_x)
    ax.set_ylabel(column_y)
    ax.set_title('Logarithmic Plot')
    
    plt.savefig(evaluation_folder + out_filename + '_high-loss.png')

    pearson_corr, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)
    print(column_x + column_y + "_high-loss," + str(pearson_corr) + "," + str(spearman_corr))
    with open(correlation_file, "a") as f:
        f.writelines(column_x + column_y + "_high-loss," + str(pearson_corr) + "," + str(spearman_corr) + "\n")
    
    #plt.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DCASE-T1 Plotting')
    parser.add_argument(
        '--results-root',
        help='Path where results are to be stored',
        required=True
    )
    
    

    args = parser.parse_args()
    correlation_file = args.results_root + "combined_evaluation/" + 'correlation.csv'
    if os.path.exists(correlation_file):
        os.remove(correlation_file)
    with open(correlation_file, "w") as f:
        f.writelines("description,pearson,spearman\n")
    plot_results(args, correlation_file, "s-adap-last_loss-test-last")
    plot_results(args, correlation_file, "s-adap-last_loss-F1-last", column_y='test_result_last', y_dict="F1")
    plot_results(args, correlation_file, "s-taylor-last_loss-test-last", column_x='train_sharpness_taylor_last')
    plot_results(args, correlation_file, "s-taylor-last_loss-F1-last", column_x='train_sharpness_taylor_last', column_y='test_result_last', y_dict="F1")
    plot_results(args, correlation_file, "loss-train_loss-F1-last", column_x='train_loss_last', column_y='test_result_last', y_dict="F1")
    
