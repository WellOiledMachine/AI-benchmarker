# AI-Benchmarker
- [📝 Overview](#-overview)
- [✨ Features](#-features)
- [🛠️ Installation](#%EF%B8%8F-installation)
- [📚 Usage](#-usage)
  - [🖼️ Image Classification Workflow Usage](#%EF%B8%8F-image-classification-workflow-usage)
    - [Options](#options)
      - [Required Options](#required-options)
      - [Dataset Specific options](#dataset-specific-options)
      - [Other options](#other-options)
      - [Using a configuration file](#using-a-configuration-file)
  - [🧩 Image Segmentation Workflow Usage](#-image-segmentation-workflow-usage)
    - [Options](#options-1)
      - [Required Options](#required-options-1)
  - [Using a configuration file](#using-a-configuration-file)
  - [Downloading Datasets](#downloading-datasets)
    - [Usage](#usage)
  - [Alpha Testing](#alpha-testing)
  - [Metrics Used in Profiling](#metrics-used-in-profiling)
    - [CPU_RAM_Utilization.csv](#cpu_ram_utilizationcsv)
    - [GPU_Utilization.csv](#gpu_utilizationcsv)
    - [training_results.csv](#training_resultscsv)
  - [Running on Consumer Hardware](#running-on-consumer-hardware)
  - [Branches](#branches)
  - [Integration with Hugging Face Ecosystem](#integration-with-hugging-face-ecosystem)
    - [Transformers Library](#transformers-library)
    - [Accelerate Library](#accelerate-library)
    - [Datasets Library](#datasets-library)
    - [Flash Attention](#flash-attention)
    - [Seamless Ecosystem Compatibility](#seamless-ecosystem-compatibility)
    - [Supplimentary Resources](#supplimentary-resources)
  - [Contribution](#contribution)

<!-- - [Using in Apptainer](#using-in-apptainer)
      - [Creating an Apptainer Container](#creating-an-apptainer-container)
      - [Running the Apptainer Container](#running-the-apptainer-container)
      - [Using Apptainer with a Configuration File](#using-apptainer-with-a-configuration-file) -->

      
# 📝 Overview
The AI Benchmarking Toolkit is a specialized tool designed to facilitate the execution of AI workloads for assessing the capabilities of various hardware configurations. It provides a framework for training and inference of diverse model archetypes, including Large Language Models, Text Classification, Object Recognition, and Voice Synthesis, while profiling the underlying hardware performance. This includes monitoring metrics such as throughput, disk I/O operations per second (IOPS), and memory usage, alongside model-specific statistics like reported loss.

# ✨ Features

- **Model Training and Inference**: Supports operations like training, pretraining, and inference with popular models such as llama-2
- **Custom Tokenizer Support**: Includes functionality to grab pre-trained tokenizers and customize them with specific tokens for tasks like padding and sequence delineation.
- **Performance Profiling**: Integrates with system hardware to log performance metrics during model operations, helping identify bottlenecks, inefficiencies, and overall system performance.
- **Data Handling**: Implements custom collate functions for data loaders to handle batches of text data, ensuring efficient processing for LLMs, Image Segmentation and Classification Models, and other models planned for future implementation.
- **Image Segmentation Training Workflow**: Empowers users to extend the capabilities of chosen models through additional training on designated datasets. It offers a versatile data loader capable of preprocessing various Hugging Face image segmentation datasets for seamless integration into the training pipeline.
- **Image Classification Training Workflow**: Enables comprehensive customization of training configurations for Image Classification models. Users have the flexibility to tailor every aspect of the training process according to their specific requirements. Additionally, the workflow provides default configurations for streamlined training of basic models, ensuring a hassle-free experience.

# 🛠️ Installation 

1. Clone the repository:
```bash
git clone https://github.com/WellOiledMachine/AI-benchmarker.git
```
2. Navigate to the project directory:
```bash
cd AI-benchmarker
```
3. Install dependencies with [Miniconda](https://docs.anaconda.com/miniconda/).
```bash
conda deactivate
conda env create -f environment.yml -y
conda activate ai-benchmarker
```
4. Done! You can deactivate this Conda environment using `conda deactivate`. 
# 📚 Usage

**Before you can use any of the workflows in this project, you must login to the HuggingFace CLI using the instructions found [here](https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login).** (Make sure to install the required packages first!)

- Some datasets will require that you to accept the agreement on the dataset's page before you can download them.

At present, only the [Image Classification](#%EF%B8%8F-image-classification-workflow-usage) and [Image Segmentation](#-image-segmentation-workflow-usage) workflows are supported through command-line execution. This means users can finetune different models using various datasets from the command-line by providing arguments or creating a configuration file for either workflow.

## 🖼️ Image Classification Workflow Usage

> [!WARNING]  
> 🧪 Beta Testing  
> This workflow is currently in the beta testing phase.  
> Your feedback is appreciated as we work to improve it!  

To use the Image Classification Workflow, first navigate to the `AI-benchmarker/` directory in your terminal. Then use the following command to start a fine-tuning session with system monitoring:

```sh
python src/train_image_classification.py [OPTIONS]
```

The Image Classification Workflow implements a transfer learning approach. It is pre-configured to utilize the [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) architecture and fine-tune it with the [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html).   

In order to use the default configuration, use the command above and only provide the `--output` option. **The default configuration should only take around 3 minutes to execute in total. Changing configuration options will affect the execution time.**

> [!NOTE]
> You may also select your own [Image Classification Dataset from HuggingFace](https://huggingface.co/datasets?task_categories=task_categories:image-classification&sort=trending), **But there are a few things you should know**:  
> * Our code **expects** the Dataset to work seamlessly with HuggingFace Hub's configuration.
>    * This basically just means that the Dataset Viewer needs to work for your selected Dataset.
> * Specifically, for Image classification, our code expects a dataset with an **Image** column, and a **Label** column.
>    * [Example of correct format](https://huggingface.co/datasets/uoft-cs/cifar100)
> * Pay attention to the information about the [Dataset Specific Options](#dataset-specific-options) 

### Options

**NOTE: Not all available command line options are listed here. To see the extensive list of available options, use the following command:**

```sh
 python src/train_image_classification.py --help
```
> [!TIP]
> **Please Note**  
> All options can instead be placed in a [configuration file](#using-a-configuration-file), instead of entered manually into the terminal each time.

#### Required Options

The only option that you MUST pass is your preferred output location directory path. Not providing this option will result in an error. This option can also be placed in a [configuration file](#using-a-configuration-file) instead.

- `--output <value>`: The location of the output folder. This folder must either be empty, or not exist yet. All output will be placed in this folder.

#### Dataset Specific options:

As stated above, while there is a pre-configured dataset and model selected for running a simple benchmark. Changing the dataset will require you to specify dataset information.

**Here are the configuration options required for selecting a dataset:**

- `--dataset <value>`: Can take one of two forms:
    1. A location of a dataset that has already been downloaded and saved to the disk (e.g. `./datasets/cifar10`) ([more information here](#downloading-datasets)). or
    2. The id of the dataset from [HuggingFace Hub](https://huggingface.co/datasets?task_categories=task_categories:image-classification&sort=trending). The dataset will be downloaded and cached using HuggingFace's [Datasets Library](https://huggingface.co/docs/datasets/en/index). **The Default ID is**: `cifar10`.

- `--image-column <value>` - **REQUIRED IF `dataset-id` IS CHANGED**: This is the title of the column that the images will be found under. The column titles can usually be found on the dataset's page on Huggingface. Unfortunately, you must provide the correct column title. Leaving this blank will give an error if you change the dataset_id from the default.  
**Example**: `python src/train_image_classification.py --image-column image --label-column label`

- `--label-column <value>` - **REQUIRED IF `dataset-id` IS CHANGED**: This is the title of the column that the labels for each will be found under. The column titles can usually be found on the dataset's page on Huggingface. Unfortunately, you must provide the correct column title. Leaving this blank will give an error if you change the dataset_id from the default.  
**Example**: `python src/train_image_classification.py --image-column image --label-column label`

- `--train-split <value>`: The name of the split that will be used for training. The default is "train", because that is the default on Huggingface. Check your selected dataset for split information.

- `--test-split <value>`: The name of the split that will be used for evaluation. If no test split is specified, the training split will be divided into new train and test splits with an 80/20 ratio.  
**Example**: `python src/train_image_classification.py --image-column image --label-column label`

- `--split-ratio <value>`: The ratio of the training split that will be used for the test split. This is only used if the `--train-split` and `--test-split` cannot be found (or weren't provided). **Default**: 0.2.

#### Other options

- `--sampling-rate <value>`: The sampling rate in seconds for recording the benchmarking metrics. **Default**: 1. The minimum value is 1.
- `--model <value>`: The name of the model to fine-tune. At the moment, only the [Image Classifaction Models from Pytorch](https://pytorch.org/vision/main/models.html#classification) can be used. Simply select which classification model to use, and then choose one of the model builders in the list.  
**For Example**: Selecting the ResNet model from the list leads you to [this page](https://pytorch.org/vision/main/models/resnet.html), and from there, you can choose the name of one of the model builders, such as 'resnet18'. Then you would use the command:  
`python src/train_image_classification.py --model resnet18`
- `--epochs <value>`: The number of epochs to fine-tune the model. The default is set to 1 to reduce training time.  
**Example**: `python src/train_image_classification.py --epochs 10` 
 
- `--batch-size <value>`: The total number of images in each batch.  **Default**: 32.
**Example**: `python src/train_image_classification.py --batch-size 64`

- `--device <value>` - **Value can either be `cuda` or `cpu`, nothing else**: The device to fine-tune the model on. Either use cuda to try to use any available gpus, or cpu to train without any gpu acceleration. **Default**: cuda  
 **Example**: `python src/train_image_classification.py --device cpu`

- `--workers <value>`: The number of data loading workers. **Default**: 1  
**Example**: `python src/train_image_classification.py --workers 8`

- `--opt <value>`: The optimizer to use during training. **Default**: sgd  
**Example**: `python src/train_image_classification.py --opt adamw`

- `--distributed`: Using this flag will set the model to be trained using multiple devices. **Default**: Disabled  
**Example**: `python src/train_image_classification.py --distributed`

## 🧩 Image Segmentation Workflow Usage

> [!WARNING]
> 🧪 Beta Testing  
> This workflow is currently in the beta testing phase.  
> Your feedback is appreciated as we work to improve it!  

To use the Image Segmentation Workflow, first navigate to the `AI-benchmarker/` directory in your terminal. Then use the following command to start a fine-tuning session with system monitoring:

```sh
python src/train_image_segmentation.py [OPTIONS]
```

The Image Segmentation Workflow implements a transfer learning approach.  
It is pre-configured to utilize the [nvidia/mit-b0](https://huggingface.co/nvidia/mit-b0) model architecture and fine-tune it with the [sidewalk-semantic](https://huggingface.co/datasets/segments/sidewalk-semantic).  


In order to use the default configuration, use the command above and only provide the `--output` option. **The default configuration should only take around 3 minutes to execute in total. Changing configuration options will affect the execution time.**

> [!CAUTION]
> **It has been found that implementing a standard Dataset Loader for [Image Segmentation Datasets](https://huggingface.co/datasets?task_categories=task_categories%3Aimage-segmentation) is a difficult task, mostly due to lack of existing dataset standards**.  
> Because of this, it is **recommended** that only those with sufficient python and AI programming knowledge try to change the dataset and model for this workflow.  
### Options
You may view all options from the command line with the following command:
```sh
 python src/train_image_classification.py --help
```

These options are explained in more detail down below.

> [!TIP]
> **Please Note**  
> All options can instead be placed in a [configuration file](#using-a-configuration-file), instead of entered manually into the terminal each time.

#### Required Options

The only option that you MUST pass is your preferred output location directory path. Not providing this option will result in an error. This option can also be placed in a [configuration file](#using-a-configuration-file) instead.

- `--output <value>`: The location of the output folder. This folder must either be empty, or not exist yet. All output will be placed in this folder.

#### Other options
- `--model <value>`: The name of the model to fine-tune. CHANGING THIS VALUE IS CURRENTLY NOT RECOMMENDED. **DEFAULT: `nvidia/mit-b0`**
- `--epochs <value>`: Defines how many times the model will be trained on the entire dataset. **DEFAULT: 1**
- `--batch-size <value>`: Specifies the number of training samples processed together in one iteration. Larger values require more memory. **DEFAULT: 10**
- `[-lr | --learning-rate] <value>`: Sets the step size at which the model adjusts its weights during training. **DEFAULT: 0.00006**
- `--device [gpu | cpu]:` Specifies whether to train on the GPU device, or on the CPU. **DEFAULT: gpu**

## Using a configuration file

Usage: `python src/train_image_classification.py --config <file_location>`

If you would rather set all of the configuration settings using a file, that is possible as well. Simply create a `JSON` file with all of the options and the values you would like to specify. 

Each of the options that can be used in a configuration file are the same as those listed in the Options sections. 

Simply omit the "--" and surround each option and string-like value in quotations.  

**For Example**: Here is a valid configuration file.

```json
{
    "output": "./test-output/",
    "model": "resnet101", 
    "epochs": 1,
    "batch-size": 32,
    "opt": "adamw",
    "dataset": "cifar10",
    "train-split": "train",
    "test-split": "test",
    "image-column": "img",
    "label-column": "label"
}
```

Notice how there are no quotations around the numerical values.

Once you have a configuration file created and saved, pass its location to the command.  
**Example**: `python src/train_image_classification.py --config config.json` 

## Downloading Datasets
We have implemented a function to download a dataset (or a subset of a datset) locally to a specified location on your disk. You may also use it to download only a fraction of a specific download.

### Usage
```bash
python src/utils.py --dataset <value> --output <value> [--fraction <value>]
```
- `--dataset <value>`: The ID of the dataset you would like to download from Huggingface.
- `--output <value>`: The path of the output directory that you would like the downloaded dataset to be placed.
- `--fraction <value>`: A float value between 0.0 and 1.0, specifying the fraction of the dataset you would like to download. 

<!-- ### Using in Apptainer

Docker image files can be converted into apptainer containers. If you wish to use apptainer instead of docker, you can follow the instructions below to create your own apptainer container from the provided docker image file.

#### Creating an Apptainer Container

1. Decompress the docker image .tar file. (If you haven't already).

```sh
tar -xvzf ai-benchmark-test.tar.gz 
```

2. Convert the docker image to an apptainer container. (This does not require having docker installed on your machine.)

```sh
apptainer build docker-archive://ai-benchmark-test.tar
```

Now you should have a new file called `ai-benchmark-test.sif` in your working directory. This is your apptainer container file.

#### Running the Apptainer Container

Apptainer containers require a bit of extra set up in order to work properly. You will need to pass it a few extra options to ensure that it runs correctly.
First, you will need to create an output folder ON THE HOST MACHINE. This is where the output from running the benchmark will go.

**Example Apptainer Command for this Project:**

```sh
apptainer exec \
  --nv  # Enables GPU support \
  --pwd /app/AI-Benchmarking-Suite  # Sets the working directory \
  --containall # Ensures the container uses strict isolation \
  --bind <HOST_OUTPUT_FOLDER>:<CONTAINER_OUTPUT_FOLDER>  # Binds the output folder \
    ai-benchmark-test.sif python src/train_image_classification.py \
      --output <CONTAINER_OUTPUT_FOLDER> 
      # (other options here. Make sure to place '\' at the end of each line except the last one.)
```

**Note: it is important that both <CONTAINER_OUTPUT_FOLDER> paths are the same.**

This command will start the container, run the specified command (the python  command), and then exit the container once the command is finished. All output will be placed in the <HOST_OUTPUT_FOLDER>.

---

#### Using Apptainer with a Configuration File

Read [Using a Configuration File](#using-a-configuration-file) for information on how to create a configuration file.

If you would like to pass a configuration file to the python command, need to also mount the configuration file into the container. This can be done by adding another `--bind` option to the command.

**Example Apptainer Command with Configuration File:**

```sh
apptainer exec \
  --nv  # Enables GPU support \
  --pwd /app/AI-Benchmarking-Suite  # Sets the working directory \
  --containall # Ensures the container uses strict isolation \
  --bind <HOST_OUTPUT_FOLDER>:<CONTAINER_OUTPUT_FOLDER>  # Binds the output folder \
  --bind <PATH_TO_CONFIG_FILE>:/app/AI-Benchmarking-Suite/config.json  # Binds the configuration file \
    ai-benchmark-test.sif python src/train_image_classification.py \
      --config config.json
``` -->

## Alpha Testing
Our Large Language Model and Text to Speech Benchmark Workflows are currently in Alpha Testing. It is recommended for you to look at the source code before using these workflows.

These workflows can be executed using the following commands:
```bash
python src/LLMsTest.py [OPTIONS]

# AND 

python src/TTSTest.py [OPTIONS]
```

## Metrics Used in Profiling

There are a number of different profiling metrics calculated by this profiler. Each calculated metric will be placed in one of three output spreadsheet files in the output directory: `CPU_RAM_Utilization.csv`, `gpustat.csv`, and `training_results.csv`. Below shows how each metric was calculated in each file:

### CPU_RAM_Utilization.csv

- **Timestamp (Unix)**: The unix timestamp for when the metrics were recorded, computed using the python [time](https://docs.python.org/3/library/time.html) library.
- **Core Time**: This is supposed to be the number of seconds that have elapsed since starting the test, but it is not accurate. (Probably needs removed.)
- **CPU Utilization (%)**:  Percentage of CPU usage for ALL system processes, measured using the [psutil](https://psutil.readthedocs.io/en/latest/) python library.
- **Thread Count**: Number of active [threading](https://docs.python.org/3/library/threading.html#threading.Thread) threads in the python interpreter. 
- **RAM Utilization (%)**: TPercentage of RAM used by the system, measured using the [psutil](https://psutil.readthedocs.io/en/latest/) python library.
- **RAM Utilization (MB)**: Amount of RAM used by the system in Megabytes, measured with the [psutil](https://psutil.readthedocs.io/en/latest/) python library.

### GPU_Utilization.csv
All of the metrics found in this file were generated using the [pynvml](https://github.com/gpuopenanalytics/pynvml) library, except for the **Timestamp (Unix)** metric.

- **Timestamp (Unix)**: The unix timestamp for when the metrics were recorded, computed using the python [time](https://docs.python.org/3/library/time.html) library.
- **GPU ID**: Identifier of the specific GPU being monitored.
- **GPU Utilization (%)**: Percentage of the GPU currently being utilized by the system.
- **GPU Memory Used (MB)**: The amount of GPU memory currently being used, measured in Megabytes.
- **GPU Memory Total (MB)**: The total VRAM of the GPU. 
- **GPU Temperature (°C)**: Temperature of the GPU in Celcius.
- **Power Usage (W)**: The total GPU power output currently being used, measured in Watts.


### training_results.csv
This file contains training information recorded for each batch used in the training process. The metrics are as follows:
- **Timestamp (Unix)**: The unix timestamp for when the metrics were recorded, computed using the python [time](https://docs.python.org/3/library/time.html) library.
- **Epoch**: The current training cycle, starting at "0". "Epoch" is a word used to describe a single complete pass of the entire training dataset to the model.
- **Batch**: The current batch being processed in the current epoch. Batches of inputs are used so that the entire dataset isn't loaded all at once.
- **Training Loss**: Measure of model performance. The criterion used to calculate this measure depends on the model type.
- **Execution Time (sec)**: The amount of time in seconds that it took to process the batch.
- **Throughput (Seq/sec)**: The number of data points in a batch processed per second. This is calculated by dividing the configured batch size by the execution time of that batch.
- **Disk Read IOPS**: The average number of disk read operations performed each second during that batch. This is calculated using the following steps:
    - Use the [psutil](https://psutil.readthedocs.io/en/latest/) python library to record the number of read operations performed on the disk both before and after the batch has been processed.
    - Subtract the 'before' read count from the 'after' read count to find out the total number of read operations that occurred while processing that batch.
    - Divide the result by the batch execution time, to caculated the average number of read operations per second for that batch.
- **Disk Write IOPS**: The average number of disk write operations perfomred each second during that batch. This is calculated the same exact way as the **Disk Read IOPS**, except that the number of write operations are used in each caculation instead of the read operations.

## Running on Consumer Hardware
This toolkit was created with the intension of profiling powerful computation systems such as High Performance Clusters (HPC's). In order to run this profiler on consumer hardware such as a laptop or other smaller commodity computers, it is recommended to choose smaller datasets (such as CIFAR10, the default), or to use options that demand less resources. This could include:
* Using a smaller batch size (`--batch-size` option). 
* Disabling distributed training (`--distributed` option, disabled by default).
* Lowering the number of workers (`--workers` option).
* Set `--device` option to `cpu`. (Will train slower, but will allow training without an Nvidia GPU).

This is not an exhaustive list, but it may help reduce the amount of resources required in a training session.

## Branches
The main branch contains the existing build and will be added to as the project develops and adds support for different model types and standards. As of May 19, 2024 the main branch supports the HuggingFace Suite for llms with extensions for text speech algorithms like soundfile 0.12.1, speechbrain 1.0.0 to support workloads. The number of necessary libraries will likely grow with supported model types and is to be expected. Any additions to the requirements list should be recorded in the requirements.txt

Secondary Branches like Data, Provide supplimentary information and records if necessary.

## Integration with Hugging Face Ecosystem

The HPC Profiler is deeply integrated with the Hugging Face ecosystem, leveraging its robust, open-source platforms and libraries to enhance AI model training and inference capabilities. Hugging Face is renowned for its comprehensive suite of tools that simplify the deployment and scaling of AI models. Below are key aspects of how our codebase utilizes the Hugging Face ecosystem:

### Transformers Library
Our profiler utilizes the [transformers](file:///home/tyson/Desktop/Contract%20Work/WELLOILEDMACHINE_LLC/hpc_profiler_library/hpc-profiler/hpc-profiler/main.py#15%2C6-15%2C6) library extensively for accessing pre-trained models and tokenizers. This integration allows users to easily fetch and deploy state-of-the-art models like Meta's Llama 3 and mistralai's Mixtral-8x22B-Instruct-v0.1 directly within their workflows.

### Accelerate Library
We employ Hugging Face's [Accelerate](file:///home/tyson/Desktop/Contract%20Work/WELLOILEDMACHINE_LLC/hpc_profiler_library/hpc-profiler/hpc-profiler/main.py#57%2C7-57%2C7) library to seamlessly optimize and distribute AI workloads across different hardware configurations without the need for detailed knowledge of the underlying frameworks. This library supports various forms of mixed precision training, helping to speed up computations and reduce memory usage effectively.

### Datasets Library
The integration with the [datasets](file:///home/tyson/Desktop/Contract%20Work/WELLOILEDMACHINE_LLC/hpc_profiler_library/hpc-profiler/hpc-profiler/DataSetsForLLM.py#7%2C6-7%2C6) library from Hugging Face enables efficient data loading, preprocessing, and postprocessing, which are crucial for handling large volumes of text data. This library provides a simple API to load and manipulate datasets in a way that is optimized for machine learning models.

### Flash Attention
While not directly integrated into every component, our system is designed to support models that utilize Flash Attention for efficient memory usage during training large transformer models. This is particularly beneficial for training with constraints on memory and computational resources.

### Seamless Ecosystem Compatibility
By building around the Hugging Face ecosystem, our profiler ensures that users have access to a broad range of tools and models that are continually updated and maintained by one of the largest communities in AI. This approach guarantees compatibility and ease of use, making it straightforward for developers to implement, train, and deploy AI models at scale.

This integration not only enhances the functionality and efficiency of our profiler but also ensures it remains at the cutting edge of AI research and application, providing users with a powerful tool for AI model development and performance analysis.


### Supplimentary Resources
The HuggingFace Transformers Library is the Backbone of this library, it streamlines model loading and data processing, while simultaneously providing access to the largest repository of opensource models and datasets currently availble on hugginghub. There are other supplimentary libraries used to extend functionality which will be linked below:

A great start to learning the backend that is extended by this library is refering to their documentation which covers a wide variety of model tasks.
- https://huggingface.co/docs/transformers/index

Regarding actually loading datasets, huggingface datasets is the most commonly used for loading opensource datasets that already have been rigourously tested. They provide these datasets in apache arrow format which optimizes the dataloading pipeline seamlessly and provides an intuitive dataloader class which provides flexible processing methods. The datasets library loader is also used to load tqdm which is a simple way to implement progress bars for time consuming processes (Batches in an Epoch for Training loops, Data preprocessing datasets with a large amount of samples.)
- https://huggingface.co/docs/datasets/index

In order to handle multiple paradigms for parallel GPU processing, this library has been built upon the HuggingFace Accelerate Library. This is currently underutilized in the library only having been applied to the large language models portion of the codebase. Even so, this implimentation is fairly rudimentary and will be extended to provide more options to the user for configurations.
- https://huggingface.co/docs/accelerate/index

Pytorch is the core machine learning backend that huggingface uses to faccilitate it's machine learning pipelines. This core dependancy is referenced throughout the codebase to instantiate learning rate schedulers and optimizers in the training loops in the workload files. It's not essential to fully understand the nuances of Pytorch itself, as huggingface abstracts it to be much easier to work with in their wrapper classes, however it is important to understand how huggingface uses things like the "ADAMW" optimizer built into pytorch for it's training loops. The documentation can be found here:
- https://pytorch.org/docs/stable/optim.html


## Contribution
Contributions are welcome. Here is the Style guide for workload generator functions that seek to impliment other model types.

### Function Naming

    Use descriptive and meaningful names for your functions, such as train_model_BERT in the provided example.
    Avoid using abbreviations or acronyms unless they are widely recognized.
    Use lowercase with underscores to separate words (e.g., train_model_bert).

### Function Docstrings

    Provide a clear and concise description of the function's purpose in the docstring.
    Specify the parameters, their types, and descriptions using the NumPy or Google docstring format.
    Mention the return value(s) and their types.
    Include any relevant notes or caveats.

### Parameter Naming

    Use descriptive names for parameters that clearly indicate their purpose.
    Avoid using single-letter names unless they are widely recognized (e.g., x, y, i, j).
    Use lowercase with underscores to separate words.
    Specify the parameter types using type hints (e.g., trainer_data: DataLoader).

### Error Handling

    Wrap the main execution of the function in a try-except block to handle exceptions.
    Log or print meaningful error messages when exceptions occur.
    Consider raising custom exceptions for specific error scenarios.

### Logging and Debugging

    Use the built-in logging module for logging messages at different levels (e.g., debug, info, warning, error).
    Include relevant information in log messages, such as function names, parameter values, and error details.
    Use print statements judiciously for debugging purposes, and remove them before finalizing the code.

### Formatting and Style

    Follow the PEP 8 style guide for Python code formatting.
    The guide can be found here: https://peps.python.org/pep-0008/
    Use 4 spaces for indentation.
    Keep lines within the recommended maximum length (usually 79 or 88 characters).
    Use consistent naming conventions for variables, functions, and modules.
    Add comments to explain complex logic or non-obvious parts of the code.

### Modularity and Reusability

    Break down the code into smaller, modular functions that perform specific tasks.
    Ensure that each function has a single responsibility and can be easily tested and reused.
    Avoid duplicating code across functions or modules.
    Consider creating separate modules or classes for different components of the AI workload (e.g., data preprocessing, model training, evaluation). I have a dedicated subfolders for dataloaders for Large language models and text to speech models. When implimenting support for another AI model type (object recognition, text to image, etc..) keep this folder structure in mind.

### Testing and Validation

    Write unit tests for individual functions to ensure they work as expected.
    Use a testing framework like unittest or pytest to organize and run the tests.
    Include test cases for different input scenarios, including edge cases and error conditions.
    Regularly run the tests to catch regressions and ensure the code's correctness.

### Documentation and Comments

    Provide clear and comprehensive documentation for the overall AI workload and its components.
    Include usage examples, expected input formats, and output formats.
    Add comments to explain the purpose and functionality of each function or block of code.
    Use docstrings to document the purpose, parameters, and return values of each function.


## License
Distributed under the MIT License. See `LICENSE` for more information.

<!-- THIS SECTION EITHER NEEDS UPDATED OR REMOVED
## Key Components
1. **Model Operations**: Functions to handle training, pretraining, and inference. These functions are designed to work with PyTorch and Hugging Face Transformers.
   - [train_model_txt_generator](file:///home/tyson/Desktop/Contract%20Work/WELLOILEDMACHINE_LLC/hpc_profiler_library/hpc-profiler/tests/test_workloads_LLMs.py#5%2C49-5%2C49)
   - [pretrain_model_txt_generator](file:///home/tyson/Desktop/Contract%20Work/WELLOILEDMACHINE_LLC/hpc_profiler_library/hpc-profiler/tests/test_workloads_LLMs.py#5%2C76-5%2C76)
   - [infer_txt_generator](file:///home/tyson/Desktop/Contract%20Work/WELLOILEDMACHINE_LLC/hpc_profiler_library/hpc-profiler/tests/test_workloads_LLMs.py#5%2C28-5%2C28)

2. **Tokenizer Utilities**: Functions to retrieve and customize tokenizers.
   - [Grab_Tokenizer](file:///home/tyson/Desktop/Contract%20Work/WELLOILEDMACHINE_LLC/hpc_profiler_library/hpc-profiler/tests/test_workloads_LLMs.py#5%2C129-5%2C129)
   - [Grab_Tokenizer_Custom](file:///home/tyson/Desktop/Contract%20Work/WELLOILEDMACHINE_LLC/hpc_profiler_library/hpc-profiler/tests/test_workloads_LLMs.py#5%2C106-5%2C106)

3. **Data Collation**: Custom collate functions to prepare data batches for model input.
   - [collate_fn_txt_generator](file:///home/tyson/Desktop/Contract%20Work/WELLOILEDMACHINE_LLC/hpc_profiler_library/hpc-profiler/tests/test_workloads_LLMs.py#5%2C145-5%2C145)
   - [collate_fn_BERT](file:///home/tyson/Desktop/Contract%20Work/WELLOILEDMACHINE_LLC/hpc_profiler_library/hpc-profiler/tests/test_workloads_LLMs.py#5%2C171-5%2C171)

4. **Performance Metrics Logging**: Utilizes system-level monitoring to log performance metrics such as throughput and disk IOPS during model training and inference.
-->
<!-- THIS IS THE OLD USAGE INFORMATION. IT NEEDS UPDATING/ REMOVED

To use the HPC Profiler, ensure you have the necessary Python packages installed, including PyTorch, Transformers, and Accelerate. The main entry point is configured in [main.py](file:///home/tyson/Desktop/Contract%20Work/WELLOILEDMACHINE_LLC/hpc_profiler_library/hpc-profiler/hpc-profiler/main.py#1%2C1-1%2C1), where models and tokenizers are set up, and training or inference tasks are initiated.

```python
# Example: Setting up and running a training session
accelerator = Accelerator(mixed_precision="bf16")
model, optimizer, scheduler, data_loader = setup_model_and_optimizer(model_name="gpt2-large")
# Start collecting Analytics at an interval of 1 second
monitor_system_utilization(benchmark_time_interval=1):
# Start Finetuning Workload
train_model_txt_generator(data_loader, model, accelerator, optimizer, scheduler, epoch_pass=10)
```

### Image Classification Example
An example Image Classification Training Workflow using our profiler can be found in [hpcProfiler/train_image_classification.py](https://gitlab.com/welloiledmachine.llc/hpc-profiler/-/blob/main/hpcProfiler/train_image_classification.py):
```py
from datasetLoaders.ImageClassData import CIFAR10Loader
from workloads.ImageClassW import train_classification_model, get_args_parser
import Profiler as Profiler

# Grab Default Arguments
arguments = get_args_parser().parse_args()

# Manually set epochs argument. This can also be set by using flags in the command line.
arguments.epochs = 3

# Load the CIFAR-10 dataset
loader = CIFAR10Loader(arguments)
train_dataloader, test_dataloader, train_sampler, num_classes = loader.load()

# Start the system profiler
monitor = Profiler.monitor_system_utilization(benchmark_time_interval=1, model_name="resnet50", custom_path="")
monitor.start_monitoring()

# Train the model
train_classification_model("resnet50", train_dataloader, test_dataloader, train_sampler, num_classes, arguments)

# Stop profiling the system.
monitor.stop_monitoring()

```
This code will produce a trained model and detailed profiling statistics that were calculated during the training.-->
<!--
## Testing
Unit tests are provided to ensure the reliability of the codebase, particularly for the model operations and data handling functions. [Currently the Unit Tests are not ready nor adequate to provide actionable diagonostic information for developers. As the codebase matures this will be corrected]

```bash
python -m unittest discover -s tests
```
-->