# Author: Tyson Limato, Cody Sloan
# project: GPU Benchmarking
# Purpose: Seamless visualization of Non-GPU statistics
# Start Date: 6/28/2023
# Last Updated:9/22/2024
import csv
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pandas as pd
import torch

def plot_cpu_and_gpu_utilization(cpu_ram_util_df, gpu_dfs, save_path):
    """
    Plot CPU utilization and GPU utilization both against time.

    Parameters
    ----------
    cpu_ram_util_df : pd.DataFrame
        The DataFrame containing the CPU and RAM utilization benchmarks.
    gpu_dfs : list of pd.DataFrame
        A list of DataFrames containing the GPU metrics for each GPU.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None
    """
    cpu_timestamps = cpu_ram_util_df['Timestamp (Unix)'].tolist()
    cpu_utilization = cpu_ram_util_df['CPU Utilization (%)'].tolist()
    if not cpu_timestamps or not cpu_utilization:
        print('ERROR: Could not extract data from file to plot CPU utilization.')
        return
    
    gpu_timestamps = [gpu_df['Timestamp (Unix)'].tolist() for gpu_df in gpu_dfs]
    
    # Calculate AVERAGE elapsed time in seconds
    all_timestamps = [cpu_timestamps] + gpu_timestamps
    zipped_timestamps = zip(*all_timestamps)
    average_timestamps = [sum(timestamps) / len(timestamps) for timestamps in zipped_timestamps]
    start_time = average_timestamps[0]
    elapsed_time = [round(t-start_time, 2) for t in average_timestamps]

    # Plot CPU Utilization
    plt.plot(elapsed_time, cpu_utilization, linestyle='-', marker='x', markersize=5, color='blue', label='CPU Utilization')

    # Plot GPU Utilization for each GPU
    colors = ['orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    for i, gpu_df in enumerate(gpu_dfs):
        gpu_utilization = gpu_df['GPU Utilization (%)'].tolist()
        plt.plot(elapsed_time, gpu_utilization, linestyle='-', marker='o', markersize=5, color=colors[i % len(colors)], label=f'GPU {i+1} Utilization')

    # Set labels, title, and legend with increased font size
    plt.xlabel('Seconds Elapsed', fontsize=14)
    plt.ylabel('Utilization (%)', fontsize=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True)

    # Save the plot as an image file
    print("Saving:", save_path)
    plt.savefig(save_path, dpi=300)
    plt.close()
    


def plot_all_columns(file_path, save_path):
    """
    THIS FUNCTION IS NOT USED ANYMORE

    Plot All Columns

    This function plots all the columns against the Timestamp in a single graph,
    and provides a data summary.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    df = pd.read_csv(file_path)

    # Extract data from the DataFrame
    timestamp = pd.to_datetime(df['Timestamp (Unix)'], unit='s')
    cpu_utilization = df['CPU Utilization (%)']
    thread_count = df['Thread Count']
    ram_utilization_percent = df['RAM Utilization (%)']
    ram_utilization_mb = df['RAM Utilization (MB)']
    

    # Plotting
    fig, ax1 = plt.subplots()

    # Plot CPU Utilization
    ax1.plot(timestamp, cpu_utilization, linestyle='-', marker='o', markersize=5, color='blue', label='CPU Utilization')
    ax1.set_ylabel('CPU Utilization')

    # Plot Thread Count
    ax1.scatter(timestamp, thread_count, marker='o', color='red', label='Thread Count', alpha=0.5, s=5)
    ax1.set_ylabel('Thread Count')

    # Create a second y-axis for RAM Utilization
    ax2 = ax1.twinx()

    # Plot RAM Utilization (%)
    ax2.plot(timestamp, ram_utilization_percent, linestyle='-', marker='o', markersize=5, color='green',
             label='RAM Utilization (%)')
    ax2.set_ylabel('RAM Utilization (%)')

    # Plot RAM Utilization (MB)
    ax2.plot(timestamp, ram_utilization_mb, linestyle='-', marker='o', markersize=5, color='orange',
             label='RAM Utilization (MB)')
    ax2.set_ylabel('RAM Utilization (MB)')

    # Set x-axis label and title
    ax1.set_xlabel('Timestamp')
    ax1.set_title('System Utilization')

    # Show legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the plot as an image file
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Data Summary
    cpu_avg = sum(cpu_utilization) / len(cpu_utilization)
    thread_avg = sum(thread_count) / len(thread_count)
    ram_percent_avg = sum(ram_utilization_percent) / len(ram_utilization_percent)
    ram_mb_avg = sum(ram_utilization_mb) / len(ram_utilization_mb)

    print('Data Summary:')
    print(f'Average CPU Utilization: {cpu_avg:.2f}')
    print(f'Average Thread Count: {thread_avg:.2f}')
    print(f'Average RAM Utilization (%): {ram_percent_avg:.2f}')
    print(f'Average RAM Utilization (MB): {ram_mb_avg:.2f}')


def plot_cpu_utilization(cpu_ram_util_df, save_path):
    """
    THIS FUNCTION IS NOT USED ANYMORE
    
    Plot CPU Utilization

    This function plots the CPU Utilization column against the timestamp,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    cpu_ram_util_df : pd.Dataframe
        The DataFrame containing the CPU and RAM utilization benchmarks.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """

    timestamps = cpu_ram_util_df['Timestamp (Unix)'].tolist()
    cpu_utilization = cpu_ram_util_df['CPU Utilization (%)'].tolist()
    if not timestamps or not cpu_utilization:
        print('ERROR: Could not extract data from file to plot CPU utilization.')
        return
    
    # Calculate elapsed time in seconds
    start_time = timestamps[0]
    elapsed_time = [round((t - start_time), 2) for t in timestamps]
    
    # Plotting
    plt.plot(elapsed_time, cpu_utilization, marker='o', markersize=5, color='blue', linestyle='-')
    plt.xlabel('Seconds Elapsed', fontsize=14)
    plt.ylabel('CPU Utilization (%)', fontsize=14)
    # plt.title('CPU Utilization')
    plt.grid(True)

    plt.savefig(save_path, dpi=300)
    plt.close()



def plot_thread_count(cpu_ram_util_df, save_path):
    """
    Plot Thread Count

    This function plots the Thread Count column against the elapsed time in seconds,
    highlighting variations in thread usage over time.

    Parameters
    ----------
    cpu_ram_util_df : pd.Dataframe
        The DataFrame containing the CPU and RAM utilization benchmarks.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    timestamps = cpu_ram_util_df['Timestamp (Unix)'].tolist()
    thread_count = cpu_ram_util_df['Thread Count'].tolist()
    if not timestamps or not thread_count:
        print('ERROR: Could not extract data from file to plot Thread Count.')
        return
    
    # Calculate elapsed time in seconds
    start_time = timestamps[0]
    elapsed_time = [round((t - start_time), 2) for t in timestamps]

    # Plotting
    plt.scatter(elapsed_time, thread_count, marker='o', color='red', label='Thread Count', alpha=0.5, s=5)
    plt.xlabel('Seconds Elapsed')
    plt.ylabel('Thread Count')
    plt.title('Thread Count')
    plt.grid(True)
    # plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ram_utilization_percent(cpu_ram_util_df, save_path):
    """
    Plot RAM Utilization (Percentage)

    This function plots the RAM Utilization in percentage against elapsed time in seconds,
    providing insights into memory usage efficiency throughout the operation.

    Parameters
    ----------
    cpu_ram_util_df : pd.Dataframe
        The DataFrame containing the CPU and RAM utilization benchmarks.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    timestamps = cpu_ram_util_df['Timestamp (Unix)'].tolist()
    ram_utilization_percent = cpu_ram_util_df['RAM Utilization (%)'].tolist()
    if not timestamps or not ram_utilization_percent:
        print('ERROR: Could not extract data from file to plot ram utilization.')
        return
    
    # Calculate elapsed time in seconds
    start_time = timestamps[0]
    elapsed_time = [round((t - start_time), 2) for t in timestamps]

    # plt.figure(figsize=(12, 8))
    # Plotting
    plt.plot(elapsed_time, ram_utilization_percent, marker='o', markersize=5, color='green', linestyle='-')
    plt.xlabel('Seconds Elapsed', fontsize=14)
    plt.ylabel('RAM Utilization (%)', fontsize=14)
    # plt.title('RAM Utilization (%)')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ram_utilization_mb(cpu_ram_util_df, save_path):
    """
    Plot RAM Utilization (MB)

    This function plots the RAM Utilization in megabytes against elapsed time
    in seconds, useful for tracking absolute memory consumption over time.

    Parameters
    ----------
    cpu_ram_util_df : pd.Dataframe
        The DataFrame containing the CPU and RAM utilization benchmarks.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    timestamps = cpu_ram_util_df['Timestamp (Unix)'].tolist()
    ram_utilization_mb = cpu_ram_util_df['RAM Utilization (MB)'].tolist()
    if not timestamps or not ram_utilization_mb:
        print('ERROR: Could not extract data from file to plot ram utilization.')
        return
    
    # Calculate elapsed time in seconds
    start_time = timestamps[0]
    elapsed_time = [round((t - start_time), 2) for t in timestamps]

    
    # Plotting
    # plt.figure(figsize=(12, 8))
    plt.plot(elapsed_time, ram_utilization_mb, marker='o', markersize=5, color='orange', linestyle='-')
    plt.xlabel('Seconds Elapsed', fontsize=14)
    plt.ylabel('RAM Utilization (MB)', fontsize=14)
    # plt.title('RAM Utilization (MB)')
    plt.grid(True)
    # Save the plot as an image file
    # plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_loss(train_results_df, save_path):
    """
    Plot Training Loss

    This function plots the Training Loss column against the batch number,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    train_results_df : pd.Dataframe
        The DataFrame containing the training results data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    timestamps = train_results_df['Timestamp (Unix)'].tolist()
    training_loss = train_results_df['Training Loss'].tolist()
    if not timestamps or not training_loss:
        print('ERROR: Could not extract data from file to plot training loss.')
        return
    
    start_time = timestamps[0]
    elapsed_time = [round((t - start_time), 2) for t in timestamps]

    # Plotting
    # plt.figure(figsize=(12, 8))
    plt.plot(elapsed_time, training_loss, marker='o', markersize=5, color='blue', linestyle='-')
    plt.xlabel('Seconds Elapsed', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    # plt.title('Training Loss')
    plt.grid(True)

    # Save the plot as an image file
    # plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_throughput(train_results_df, save_path):
    """
    THIS FUNCTION IS NOT USED ANYMORE
    
    Plot Throughput

    This function plots the Throughput (Seq/sec) column against the batch number,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    train_results_df : pd.Dataframe
        The DataFrame containing the training results data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    batch_number = train_results_df['Batch'].tolist()
    throughput = train_results_df['Throughput (Seq/sec)'].tolist()
    if not batch_number or not throughput:
        print('ERROR: Could not extract data from file to plot throughput.')
        return

    # Plotting: COULD USE AN UPDATE
    plt.scatter(batch_number, throughput, marker='o', color='green', label='Throughput', alpha=0.5, s=5)
    plt.xlabel('Batch Number')
    plt.ylabel('Throughput (Seq/sec)')
    # plt.title('Throughput (Seq/sec)')
    plt.grid(True)
    
    # Save the plot as an image file
    # plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_disk_iops(train_results_df, save_path):
    """
    Plot Disk IOPS

    This function plots the Disk Read IOPS and Disk Write IOPS columns against the batch number on the same graph,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    train_results_df : pd.Dataframe
        The DataFrame containing the training results data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """

    timestamps = train_results_df['Timestamp (Unix)'].tolist()
    disk_read_iops = train_results_df['Disk Read IOPS'].tolist()
    disk_write_iops = train_results_df['Disk Write IOPS'].tolist()
    if not timestamps or not disk_read_iops or not disk_write_iops:
        print('ERROR: Could not extract data from file to plot disk IOPS.')
        return

    # Calculate elapsed time in seconds
    start_time = timestamps[0]
    elapsed_time = [round((t - start_time), 2) for t in timestamps]

    # Plotting
    plt.figure(figsize=(12, 8))
    font_size = 18
    plt.plot(elapsed_time, disk_read_iops, marker='o', markersize=8, color='blue', label='Disk Read IOPS', linestyle='-')
    plt.plot(elapsed_time, disk_write_iops, marker='x', markersize=8, color='orange', label='Disk Write IOPS', linestyle='-')
    plt.xlabel('Seconds Elapsed', fontsize=font_size)
    plt.ylabel('Disk IOPS', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_disk_read_iops(train_results_df, save_path):
    """
    Plot Disk Read IOPS

    This function plots the Disk Read IOPS column against the batch number,
    and incorporates various enhancements to make the graph more informative.

    Parameters
    ----------
    train_results_df : pd.Dataframe
        The DataFrame containing the training results data.
    save_path : str
        The path to save the image file.

    Returns
    -------
    None

    """
    batch_number = train_results_df['Batch'].tolist()
    disk_read_iops = train_results_df['Disk Read IOPS'].tolist()
    if not batch_number or not disk_read_iops:
        print('ERROR: Could not extract data from file to plot Disk Read IOPS.')
        return

    # Plotting
    plt.scatter(batch_number, disk_read_iops, linestyle='-', marker='o', color='purple',
                label='Disk Read IOPS', alpha=0.5, s=5)
    plt.xlabel('Batch Number')
    plt.ylabel('Disk Read IOPS (I/O Operations per Second)')
    # plt.title('Disk Read IOPS')
    plt.grid(True)

    # Save the plot as an image file
    # plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_disk_write_iops(train_results_df, save_path):
    """
    Plot Disk Write IOPS

    This function reads disk write IOPS (Input/Output Operations Per Second)
    data from a CSV file and generates a scatter plot of these values 
    against batch numbers. It also calculates and plots a moving average of 
    the IOPS to provide a smoother trend over time. The plot is then saved to a specified file path.

    Parameters
    ----------
    train_results_df : pd.Dataframe
        The DataFrame containing the training results data.
    save_path : str
        The path where the generated plot image will be saved.

    Processes
    ----------
    - Opens the CSV file and reads the data, skipping the first two header rows.
    - Extracts batch numbers and disk write IOPS values from the CSV file.
    - Plots the raw disk write IOPS data as a scatter plot.
    - Calculates a moving average of the disk write IOPS with a predefined window size and plots this on the same graph.
    - Displays a horizontal line representing the average disk write IOPS across all batches.
    - Saves the plot to an image file at the specified path.

    Returns
    -------
    None

    Notes
    -----
    - The function assumes the CSV file is contains Batch numbers and Disk write IOPS as specific column names.
    - The CSV file is expected to have one header row which is skipped during processing.
    - The plot includes labels for axes, a title, a legend, and grid lines for better readability.
    - The function handles file reading errors by ignoring them, which may not always be desirable. Adjust the error handling as necessary based on the expected data quality.
    """
    batch_number = train_results_df['Batch'].tolist()
    disk_write_iops = train_results_df['Disk Write IOPS'].tolist()
    if not batch_number or not disk_write_iops:
        print('ERROR: Could not extract data from file to plot Disk Write IOPS.')
        return

    # Plotting
    plt.scatter(batch_number, disk_write_iops, color='orange', label='Disk Write IOPS', linestyle='-', alpha=0.5, s=5)
    plt.xlabel('Batch Number')
    plt.ylabel('Disk Write IOPS')
    # plt.title('Disk Write IOPS')
    plt.grid(True)

    # Save the plot as an image file
    # plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def get_number_of_gpus(debug=False):
    """
    Get Number of CUDA-enabled GPUs

    This function retrieves the number of CUDA-enabled GPUs available on the system using PyTorch's utilities. It can optionally print the count for debugging purposes.

    Parameters
    ----------
    debug : bool, optional
        If set to True, the function will print the number of CUDA-enabled GPUs. Default is False.

    Returns
    -------
    int or None
        Returns the number of CUDA-enabled GPUs if successful, or None if an error occurs.

    Processes
    ----------
    - Utilizes PyTorch's `torch.cuda.device_count()` to get the count of CUDA-enabled GPUs.
    - If the debug parameter is True, prints the number of GPUs.
    - Handles exceptions by printing an error message and returning None.

    Notes
    -----
    - This function requires PyTorch to be installed and properly configured with CUDA support.
    - It is useful for dynamically adjusting resource allocation based on the number of available GPUs.
    - The function handles exceptions gracefully, making it robust for use in environments where CUDA may not be available or properly configured.
    """
    try:
        # Get the list of CUDA-enabled devices
        cuda_devices = torch.cuda.device_count()
        if debug:
            print(f"Number of CUDA-enabled GPUs: {cuda_devices}")
        return cuda_devices
    except Exception as e:
        print(f"Error: {e}")
        return None


def process_gpu_log(logfile, gpu_nums):
    """
    Process GPU Log Data

    This function parses a log file containing GPU metrics and organizes the data into structured dataframes for each GPU. 
    It extracts metrics such as temperature, GPU utilization, power usage, and memory usage, and timestamp info.

    Parameters
    ----------
    logfile : str
        The path to the log file to be processed.
    gpu_nums : int
        The total number of GPUs that data is logged for.

    Returns
    -------
    gpu_dfs : list of pandas.DataFrame
        A list of dataframes, where each dataframe contains the processed log data for one GPU.
    """
    
    df = pd.read_csv(logfile)
    
    gpu_dfs = []
    for gpu_id in range(gpu_nums):
        gpu_df = df[df['GPU ID'] == gpu_id].reset_index(drop=True)
        gpu_dfs.append(gpu_df)

    return gpu_dfs

    


def plot_gpu_utilization(gpu_dfs, log_path):
    """
    Plot GPU Utilization Metrics

    This function generates multiple plots for GPU utilization metrics including utilization percentage, power usage, memory usage, and temperature. Each metric is plotted separately for each GPU in the system, and the plots are saved to specified files.

    Parameters
    ----------
    gpu_dfs : list of pandas.DataFrame
        A list of dataframes where each dataframe contains the GPU metrics for a single GPU. Each dataframe must have columns 'Timestamp', 'GPU Utilization (%)', 'Power Usage (W)', 'Memory Usage (MB)', and 'Temperature (°C)'.
    log_path : str
        The directory path where the plot images will be saved.

    Processes
    ----------
    - For each metric (GPU Utilization, Power Usage, Memory Usage, Temperature), a separate plot is created:
        1. GPU Utilization: Plots the utilization over time for each GPU.
        2. Power Usage: Plots the power usage over time for each GPU.
        3. Memory Usage: Plots the memory usage over time for each GPU.
        4. Temperature: Plots the temperature over time for each GPU.
    - Each plot includes:
        - A scatter plot of the metric over time.
        - A horizontal line indicating the average value of the metric.
        - Proper labeling and a legend.
    - Each plot is saved as a PNG file in the specified log path with a name that includes the model name and the metric.

    Returns
    -------
    None

    Notes
    -----
    - Ensure that the 'log_path' directory exists and is writable.
    - The function assumes that the dataframes in 'gpu_dfs' are properly formatted and contain the necessary columns.
    - This function does not display the plots but saves them directly to files. If visualization is needed during debugging or analysis, consider uncommenting the plt.show() line at the end of the function.
    """
    gpu_nums = len(gpu_dfs)

    # Create a figure for GPU Utilization
    plt.figure(figsize=(12, 8))
    
    for i, gpu_df in enumerate(gpu_dfs):
        timestamps = gpu_df['Timestamp (Unix)'].tolist()
        gpu_utilization = gpu_df['GPU Utilization (%)'].tolist()
        if not timestamps or not gpu_utilization:
            print('ERROR: Could not extract data from file to plot gpu utilzation.')
            return
        
        # Calculate elapsed time in seconds
        start_time = timestamps[0]
        elapsed_time = [round((t - start_time), 2) for t in timestamps]

        plt.subplot(1, gpu_nums, i + 1)
        plt.plot(elapsed_time, gpu_utilization, marker='o', markersize=5, linestyle='-', label=f'GPU {i + 1}')
        plt.xlabel('Seconds Elapsed', fontsize=14)
        plt.ylabel('GPU Utilization (%)', fontsize=14)
        plt.ylim(-5, 100)
        plt.legend()

    plt.tight_layout()
    # Save the figure to a file
    plt.savefig(os.path.join(log_path,'gpu_utilization.png'))
    plt.close()

    return # We no longer care about anything other than gpu utilization

    # Create a separate figure for Power Usage
    plt.figure(figsize=(7 * gpu_nums, 7))
    plt.suptitle(f'GPU Power Utilization (spec power 165W) ({gpu_nums} GPUs) | Model: {model_name}')

    for i, gpu_df in enumerate(gpu_dfs):
        plt.subplot(1, gpu_nums, i + 1)
        plt.scatter(gpu_df['Timestamp'] / 60, gpu_df['Power Usage (W)'], label=f'GPU {i + 1}', alpha=0.5, s=5)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Power Usage (W)')
        plt.ylim(-5, 400)
        plt.axhline(y=gpu_df['Power Usage (W)'].mean(), color='r', linestyle='--', label='Average Power Usage')
        plt.legend()

    plt.tight_layout()
    # Save the figure to a file
    plt.savefig(os.path.join(log_path,'gpu_power_usage.png'))
    plt.close()

    # Create a separate figure for Memory Utilization
    plt.figure(figsize=(7 * gpu_nums, 7))
    plt.suptitle(
        f'GPU Memory Utilization (total memory 24576 MB) ({gpu_nums} GPUs) | Model: {model_name}')

    for i, gpu_df in enumerate(gpu_dfs):
        plt.subplot(1, gpu_nums, i + 1)
        plt.scatter(gpu_df['Timestamp'] / 60, gpu_df['Memory Usage (MB)'], label=f'GPU {i + 1}', alpha=0.5, s=5)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Memory Usage (MB)')
        plt.ylim(-100, 24576)
        plt.axhline(y=gpu_df['Memory Usage (MB)'].mean(), color='r', linestyle='--', label='Average Memory Usage (MB)')
        plt.legend()

    plt.tight_layout()
    # Save the figure to a file
    plt.savefig(os.path.join(log_path,'gpu_memory_usage.png'))
    plt.close()

    # Create a separate figure for Temperature
    plt.figure(figsize=(7 * gpu_nums, 7))
    plt.suptitle(f'GPU Temperature (°C) ({gpu_nums} GPUs) | Model: {model_name}')

    for i, gpu_df in enumerate(gpu_dfs):
        plt.subplot(1, gpu_nums, i + 1)
        plt.scatter(gpu_df['Timestamp'] / 60, gpu_df['Temperature (°C)'], label=f'GPU {i + 1}', alpha=0.5, s=5)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Temperature (°C)')
        plt.ylim(-5, 100)
        plt.axhline(y=gpu_df['Temperature (°C)'].mean(), color='r', linestyle='--', label='Average Temperature')
        plt.legend()

    plt.tight_layout()
    # Save the figure to a file
    plt.savefig(os.path.join(log_path,'gpu_temperature.png'))
    plt.close()

    # Display the entire plot
    # plt.show()


# PREPROCESSING FUNCTIONS INCASE OF OVER THE AIR TRANSFER ERRORS WITH KUBECTL

def remove_nul_from_csv(input_file, output_file):
    """
    Remove NUL Characters from CSV File

    This function processes a CSV file to remove all NUL (null) characters from the data. NUL characters can cause issues during file processing and data analysis. The function reads the original CSV file, removes any NUL characters, and writes the cleaned data back to a new file. This helps ensure data integrity and compatibility with various data processing tools.

    Parameters
    ----------
    input_file : str
        The path to the CSV file from which NUL characters will be removed.
    output_file : str
        The path to the new CSV file where the cleaned data will be written.

    Processes
    ----------
    - Opens the specified input CSV file in read mode with UTF-8 encoding and replaces errors.
    - Removes all NUL characters from the data.
    - Writes the cleaned data to the specified output CSV file in write mode with UTF-8 encoding.

    Returns
    -------
    None

    Notes
    -----
    - This function writes the cleaned data to a new file rather than modifying the original file directly, allowing for data preservation if needed.
    - Ensure that the output file path is accessible and writable to avoid file writing errors.
    - Removing NUL characters is particularly important when dealing with data exported from systems that may insert these characters as placeholders or data padding.
    """
    with open(input_file, 'r', newline='', encoding='utf-8', errors='replace') as csv_file:
        data = csv_file.read()

    # Remove NUL characters from the data
    clean_data = data.replace('\x00', '')

    # Write the clean data to a new CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
        out_file.write(clean_data)


def remove_non_ascii(input_file, output_file):
    """
    Remove Non-ASCII Characters from CSV File

    This function processes a CSV file to remove all non-ASCII characters from the data. It reads the original CSV file, filters out any characters that are not within the ASCII range, and writes the cleaned data back to a new file. This is useful for standardizing file encoding and avoiding issues with character encoding compatibility.

    Parameters
    ----------
    input_file : str
        The path to the CSV file from which non-ASCII characters will be removed.
    output_file : str
        The path to the new CSV file where the cleaned data will be written.

    Processes
    ----------
    - Opens the specified input CSV file in read mode with UTF-8 encoding and replaces errors.
    - Filters out all non-ASCII characters (i.e., characters with a code point greater than 127).
    - Writes the cleaned data to the specified output CSV file in write mode with UTF-8 encoding.

    Returns
    -------
    None

    Notes
    -----
    - This function writes the cleaned data to a new file rather than modifying the original file directly, allowing for data preservation if needed.
    - Ensure that the output file path is accessible and writable to avoid file writing errors.
    - This function can be particularly useful when dealing with data that needs to be compatible with systems that only support ASCII character encoding.
    """
    with open(input_file, 'r', newline='', encoding='utf-8', errors='replace') as csv_file:
        data = csv_file.read()

    # Remove non-ASCII characters from the data
    clean_data = ''.join(char for char in data if ord(char) < 128)

    # Write the clean data to a new CSV file with proper encoding
    with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
        out_file.write(clean_data)


def remove_redundant_headers(file_path):
    """
    Remove Redundant Headers from CSV File

    This function identifies and removes redundant header rows from a CSV file. It is particularly useful for cleaning up files that have accumulated multiple header rows due to concatenation or other data processing errors.

    Parameters
    ----------
    file_path : str
        The path to the CSV file that needs header cleanup. The file will be modified in place.

    Processes
    ----------
    - Opens the specified CSV file in read mode and reads all data into a list of rows.
    - Searches for the first occurrence of a header row, which is identified by searching for the keyword "Timestamp (Unix)".
    - Removes any subsequent occurrences of the header row found after the first occurrence.
    - Writes the cleaned data back to the CSV file in write mode.

    Returns
    -------
    None

    Notes
    -----
    - This function modifies the original file directly, so ensure that a backup is made if original data preservation is necessary.
    - The function assumes that the CSV file is properly formatted and that the header row can be identified by a specific keyword.
    - Care should be taken to adjust the keywords if the header format changes or if different files have different header contents.
    """
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

    # Identify the first header row
    header_found = False
    cleaned_data = []

    for row in data:
        if 'Timestamp (Unix)' in row[0]:
            if not header_found:
                header_found = True
                cleaned_data.append(row)
        else:
            cleaned_data.append(row)

    # Write the updated data back to the file
    with open(file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(cleaned_data)


def clean_strings_quotes_from_csv(file_path):
    """
    Clean Quotes from CSV Strings

    This function processes a CSV file to remove double quotes from the beginning and end of each field in every row. It reads the entire CSV file into memory, processes each row to strip the quotes, and writes the cleaned data back to the same file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file that needs cleaning. The file will be modified in place.

    Processes
    ----------
    - Opens the specified CSV file in read mode and reads all data into a single string.
    - Splits the data string into individual rows based on newline characters.
    - Removes double quotes from the start and end of each field in each row.
    - Joins the cleaned rows back into a single string with newline characters.
    - Writes the cleaned data back to the CSV file in write mode.

    Returns
    -------
    None

    Notes
    -----
    - This function modifies the original file directly, so ensure that a backup is made if original data preservation is necessary.
    - The function assumes that the CSV file is properly formatted and that fields may be enclosed in double quotes.
    - Care should be taken with fields that may intentionally contain double quotes as part of the field data, as this function will remove them if they are at the start or end of the field.
    """
    with open(file_path, mode='r', newline='') as file:
        data = file.read()

    # Split the data by newline characters to get individual rows
    rows = data.strip().split('\n')

    # Remove double-quote characters from both ends of each value in each row
    cleaned_rows = []
    for row in rows:
        cleaned_row = ','.join(value.strip('"') for value in row.split(','))
        cleaned_rows.append(cleaned_row)

    # Join the cleaned rows back into a single string with newline characters
    cleaned_data = '\n'.join(cleaned_rows)

    # Write the cleaned data back to the file
    with open(file_path, mode='w', newline='') as file:
        file.write(cleaned_data)


def remove_quotes_from_csv(file_path):
    """
    Remove Quotes from CSV File

    This function processes a CSV file to remove any double quotes surrounding the entries in each row. It reads the original CSV file, strips the quotes from each field, and writes the cleaned data back to the same file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file that needs quote removal. The file will be modified in place.

    Processes
    ----------
    - Opens the specified CSV file in read mode and reads all data into memory.
    - Processes each row to remove double quotes from around each field.
    - Opens the CSV file again in write mode and writes the cleaned data back to the file.

    Returns
    -------
    None

    Notes
    -----
    - This function modifies the original file directly, so ensure that a backup is made if original data preservation is necessary.
    - The function assumes that the CSV file is properly formatted and that each field is potentially enclosed in double quotes.
    - Care should be taken with fields that may intentionally contain double quotes as part of the field data.
    """
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

    with open(file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        for row in data:
            cleaned_row = [value.strip('"') for value in row]
            csv_writer.writerow(cleaned_row)


def read_last_n_rows(file_path, n):
    """
    Read Last N Rows from a CSV File

    This function reads the last N rows from a CSV file, which can be particularly useful for analyzing or displaying the most recent data entries. It returns both the headers and the last N rows of the file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file from which data will be read.
    n : int
        The number of rows to read from the end of the file.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - headers (list): The column headers of the CSV file.
        - last_n_rows (list of lists): The last N rows of data from the CSV file.

    Notes
    -----
    - This function loads the entire file into memory, which may not be efficient for very large files.
    - Ensure that the file exists and is accessible to avoid runtime errors.
    - The function assumes that the first row of the CSV file contains the headers.
    """
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)
        headers = data[0]
        last_n_rows = data[-n:]
        return headers, last_n_rows


def write_to_csv(file_path, headers, data):
    """
    Write Data to CSV File

    This function writes a list of data rows to a CSV file, including headers. It is designed to handle the creation or overwriting of CSV files with specified data, ensuring that the data is formatted correctly for CSV storage.

    Parameters
    ----------
    file_path : str
        The path to the CSV file where the data will be written. If the file already exists, it will be overwritten.
    headers : list
        A list of strings representing the column headers for the CSV file.
    data : list of lists
        A list where each sublist represents a row of data that corresponds to the headers.

    Processes
    ----------
    - Opens the specified file in write mode.
    - Creates a CSV writer object to handle the writing of data.
    - Writes the headers to the first row of the CSV file.
    - Writes subsequent rows of data under the headers.
    - Closes the file after writing is complete.
    - Prints a confirmation message indicating successful data writing.

    Returns
    -------
    None

    Notes
    -----
    - This function will overwrite existing files without warning. Ensure that this behavior is intended, or implement checks before writing to existing files.
    - Proper error handling should be implemented outside this function to manage issues like permissions or disk space errors during file operations.
    """
    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(headers)
        csv_writer.writerows(data)
    print(f"Data successfully written to {file_path}.")


def preprocess_data(InputDirectory: str):
    """
    Preprocess Data for a Given Model

    This function performs a series of data cleaning operations on CSV files related to a specific model. It is designed to ensure that the data is free of common formatting issues before further processing or analysis. The function handles tasks such as removing null bytes, non-ASCII characters, redundant headers, and unnecessary quotes from the data files.

    Parameters
    ----------
    InputDirectory : str
        The directory containing the data files to be preprocessed. This name is used to construct paths to the relevant CSV files.

    Processes
    ----------
    - Retrieves the project directory path.
    - Removes null bytes from the CSV files to prevent errors during file reading.
    - Removes non-ASCII characters to standardize the file encoding.
    - Removes redundant headers that might have been mistakenly included multiple times.
    - Cleans up strings in the CSV files by removing unnecessary quotes around entries.
    - Each cleaning step is applied to both the training results and CPU/RAM utilization data files for the specified model.

    Returns
    -------
    None

    Notes
    -----
    - The function assumes that the CSV files are located in a structured directory format under the project directory.
    - It is crucial that the CSV files are backed up before performing these operations, as the changes are written directly to the original files.
    - The function prints a message before and after the preprocessing steps to provide feedback on the operation's progress.
    """
    project_directory = get_project_directory()
    remove_nul_from_csv(os.path.join(InputDirectory, 'training_results.csv'),
                        os.path.join(InputDirectory, 'training_results.csv'))
    remove_nul_from_csv(os.path.join(InputDirectory, 'CPU_RAM_Utilization.csv'),
                        os.path.join(InputDirectory, 'CPU_RAM_Utilization.csv'))

    remove_non_ascii(os.path.join(InputDirectory, 'training_results.csv'),
                     os.path.join(InputDirectory, 'training_results.csv'))
    remove_non_ascii(os.path.join(InputDirectory, 'CPU_RAM_Utilization.csv'),
                     os.path.join(InputDirectory, 'CPU_RAM_Utilization.csv'))

    remove_redundant_headers(os.path.join(InputDirectory, 'training_results.csv'))
    remove_redundant_headers(os.path.join(InputDirectory, 'CPU_RAM_Utilization.csv'))

    clean_strings_quotes_from_csv(os.path.join(InputDirectory, 'training_results.csv'))
    clean_strings_quotes_from_csv(os.path.join(InputDirectory, 'CPU_RAM_Utilization.csv'))

    remove_quotes_from_csv(os.path.join(InputDirectory, 'training_results.csv'))
    remove_quotes_from_csv(os.path.join(InputDirectory, 'CPU_RAM_Utilization.csv'))


def get_project_directory():
    """
    Get Project Directory

    This function retrieves the absolute path of the project directory by navigating up from the current script's directory. It is useful for constructing paths that are relative to the project's root directory, ensuring that file operations are executed in the correct context.

    Processes
    ----------
    - Determines the directory of the currently executing script.
    - Navigates up two levels from the current directory to reach the project's root directory, assuming the script is located behind two levels of directories from the repository root.

    Returns
    -------
    str
        The absolute path to the project directory.

    Notes
    -----
    - This function assumes a specific directory structure. If the directory structure of the project changes, the function may need to be updated to reflect these changes.
    """
    # Get the absolute path of the script being executed
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # Navigate up two levels to get the repository root directory
    project_directory = os.path.abspath(os.path.join(current_dir, '..', '..'))
    return project_directory

def generate_graphs(InputDirectory: str, GraphDirectory: str):
    """
    Generate Graphs for Model Analysis

    This function orchestrates the generation of various performance and utilization graphs for a given model. It handles the plotting of training loss, throughput, disk I/O operations, CPU utilization, thread count, RAM utilization, and GPU utilization. The function retrieves data from specified CSV files and logs, processes them, and saves the resulting plots to a designated directory.

    Parameters
    ----------
    InputDirectory : str
        The directory of saved model logs and checkpoints. This name is used to locate the appropriate data files and to name the output files.
    GraphDirectory : str
        The directory within the project where the generated graph images will be saved. This directory is relative to the project directory.
    
    Processes
    ----------
    - Retrieves the project directory path.
    - Generates and saves the following graphs:
        * Training loss
        * Throughput
        * Disk read IOPS
        * Disk write IOPS
        * Combined disk IOPS
        * CPU utilization percentage
        * Thread count
        * RAM utilization in percentage
        * RAM utilization in MB
        * GPU utilization (if applicable)
    - Each graph is saved using the model name and the type of graph as the file name.
    - For GPU utilization, processes GPU log data and generates a utilization graph per GPU.

    Returns
    -------
    None

    Notes
    -----
    - Ensure that the CSV and log files are correctly formatted and accessible.
    - The function assumes the presence of specific columns in the CSV files for each metric.
    """
    os.makedirs(os.path.join(InputDirectory, GraphDirectory), exist_ok=True)
    training_results = pd.read_csv(os.path.join(InputDirectory, 'training_results.csv'))
    cpu_ram_utilization = pd.read_csv(os.path.join(InputDirectory, 'CPU_RAM_Utilization.csv'))

    
    print("Generating Graphs...")
    plot_training_loss(training_results,
                       os.path.join(InputDirectory, GraphDirectory, 'training_loss.png'))
    # plot_throughput(training_results,
    #                 os.path.join(InputDirectory, GraphDirectory, 'throughput.png'))
    # plot_disk_read_iops(training_results,
    #                     os.path.join(InputDirectory, GraphDirectory, 'disk_read_iops.png'))
    # plot_disk_write_iops(training_results,
    #                      os.path.join(InputDirectory, GraphDirectory, 'disk_write_iops.png'))
    plot_disk_iops(training_results,
                   os.path.join(InputDirectory, GraphDirectory, 'disk_iops.png'))
    plot_cpu_utilization(cpu_ram_utilization,
                         os.path.join(InputDirectory, GraphDirectory, 'cpu_utilization_percent.png'))
    # plot_thread_count(cpu_ram_utilization,
    #                   os.path.join(InputDirectory, GraphDirectory, 'thread_count.png'))
    plot_ram_utilization_percent(cpu_ram_utilization,
                                 os.path.join(InputDirectory, GraphDirectory, 'ram_utilization_percent.png'))
    plot_ram_utilization_mb(cpu_ram_utilization,
                            os.path.join(InputDirectory, GraphDirectory, 'ram_utilization_mb.png'))

    # GPUS
    gpu_logfile = os.path.join(InputDirectory, 'GPU_Utilization.csv')
    gpu_nums = get_number_of_gpus(debug=False)
    gpu_dfs = process_gpu_log(open(gpu_logfile, 'r'), gpu_nums)
    log_path = os.path.join(InputDirectory, GraphDirectory)
    plot_gpu_utilization(gpu_dfs, log_path)
    plot_cpu_and_gpu_utilization(cpu_ram_utilization, gpu_dfs, os.path.join(InputDirectory, GraphDirectory,"cpu_gpu_utilization.png"))


# Function to measure execution time
def measure_time(func):
    """
    Measure Execution Time of a Function

    This function measures the execution time of a given function. It is useful for performance testing and optimization by providing the time taken to execute the function from start to finish.

    Parameters
    ----------
    func : callable
        The function whose execution time is to be measured. This function should not require any arguments.

    Processes
    ----------
    - Records the start time before the function is called.
    - Executes the function.
    - Records the end time after the function execution completes.
    - Calculates the execution time by subtracting the start time from the end time.
    - Returns the execution time in seconds, formatted to two decimal places.

    Returns
    -------
    None

    Notes
    -----
    - The function passed to `measure_time` should not take any arguments; if the function requires arguments, consider using `functools.partial` or a lambda function to pre-specify them before measurement.
    - This function is designed for simplicity and may not handle functions that are asynchronous or involve multi-threading/multi-processing.
    """
    start_time = time.time()
    func()
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    return execution_time


def preprocess_and_generate_graphs(InputDirectory: str, GraphDirectory: str):
    """
    Preprocess and Generate Graphs for a Model

    This function preprocesses the data for a given model and then generates the performance and utilization graphs.
    """
    # Set default font size for plots
    preprocess_data(InputDirectory)
    generate_graphs(InputDirectory, GraphDirectory)


# Measure the time it takes to preprocess and generate graphs
def all_inclusive_plots(InputDirectory: str, GraphDirectory: str):
    """
    Generate All-Inclusive Graphs for a Model

    This function calls the preprocess_and_generate_graphs function and measures the time taken to complete.
    """
    execution_time = measure_time(lambda: preprocess_and_generate_graphs(InputDirectory, GraphDirectory))
    print(f"Graphs generated successfully. Took {execution_time:.2f} seconds.")



if __name__ == "__main__":
    InputDirectory = "classif"
    GraphDirectory = "graphs"

    all_inclusive_plots(InputDirectory, GraphDirectory)

    # os.makedirs(os.path.join(InputDirectory, GraphDirectory), exist_ok=True)
    # cpu_ram_utilization = pd.read_csv(os.path.join(InputDirectory, 'CPU_RAM_Utilization.csv'))

    # gpu_logfile = os.path.join(InputDirectory, 'gpustat.csv')
    # gpu_nums = get_number_of_gpus(debug=False)
    # gpu_dfs = process_gpu_log(open(gpu_logfile, 'r'), gpu_nums)
    # log_path = os.path.join(InputDirectory, GraphDirectory)
    # plot_gpu_utilization(gpu_dfs, log_path)
    # plot_cpu_and_gpu_utilization(cpu_ram_utilization, gpu_dfs, os.path.join(InputDirectory, GraphDirectory,"cpu_gpu_utilization.png"))

    # remove_redundant_headers("graphtest/gpustat.csv")