# Authors: Tyson Limato, Cody Sloan
# project: AI Benchmarker
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch

# High level Imports
import csv
import os
from accelerate import accelerator
from accelerate import FullyShardedDataParallelPlugin
from Profiler import monitor_system_utilization
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from torch.optim import AdamW, lr_scheduler
# Hugging Face Imports
from workloads.LLMsW import create_and_prepare_GPTQ_model, get_project_directory
from graphingScripts.PlotFunctions import all_inclusive_plots
from datasetLoaders.LLMsData import OpenWebTextDataset
from workloads.LLMsW import train_model_txt_generator, infer_txt_generator, get_project_directory
import torch.quantization

def get_parsed_args(add_help=True):
    """
    Parse the command line arguments
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="PyTorch GPT2 Training",
        add_help=add_help,
    )
    parser.add_argument("--output", type=str, required=True, help="Output Directory")
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2", help="Model Name")
    parser.add_argument("--data", type=str, default="c4", help="Dataset Name")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA Alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA Dropout")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA R")
    parser.add_argument("--precision_type", type=str, default="bf16", help="Precision Type")
    parser.add_argument("--benchmark_time_interval", type=int, default=1, help="Benchmark Time Interval")

    return parser.parse_args()


def main():
    # Environment Defined training Parameters
    # Define the arguments for creating and preparing the GPTQ model
    args = get_parsed_args()
    # Print the chosen arguments
    print(args)

    proj_dir = get_project_directory()

    processed_model_name = args.model_name.replace('/', '-').replace(':', '-')
    output_file_path = os.path.join(args.output, f'{processed_model_name}_model_output.csv')

    # data = 'c4'
    # lora_alpha = 16
    # lora_dropout = 0.1
    # lora_r = 64
    # Ben_batch_size = 4
    # precision_type = "bf16"
    # task_type = "finetune"
    # benchmark_time_interval = 1

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        state_dict_type=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
        sharding_strategy="FULL_SHARD"
    )
    accel = accelerator.Accelerator(mixed_precision=args.precision_type, fsdp_plugin=fsdp_plugin)

    # Optionally, you can define a device map if you have specific GPU configurations
    device_map = None  # or specify a device map, e.g., {"": 0} for single GPU

    # Call the function to create and prepare the GPTQ model
    model, peft_config, tokenizer = create_and_prepare_GPTQ_model(
        model_name=args.model_name,
        dataset_name=args.data,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_r=args.lora_r,
        device_map=device_map,
        output_dir=args.output
    )
    print("Model, PEFT config, and tokenizer are ready and model is quantized.")


    TrainChatData = OpenWebTextDataset(tokenizer=tokenizer, split='train')
    optim = AdamW(model.parameters(), lr=0.0001)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=0.0001)
    # Accelerate Distributed passthrough
    model, optim, scheduler, TrainChatData = accel.prepare(model, optim, scheduler, TrainChatData)
    monitor=monitor_system_utilization(interval=args.benchmark_time_interval, output_dir=args.output)
    monitor.start_monitoring()
    try:
        # Call Training Function (Will write a CSV file)
        epoch = int(os.environ.get('NUM_EPOCHS')) if os.environ.get('NUM_EPOCHS') is not None else 1
        # Set Token length per Text Entry
        # (Entries Longer than specified number will be truncated and Entries Shorter will be Padded with empty tokens)
        # GPT2 has a max length of 1024 tokens
        # According to OpenAI, the conversion rate of character to token is 4:1
        # Cite: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them

        # Training RunTime
        print("Fine-tuning...")
        train_model_txt_generator(args.model_name, TrainChatData, model, accel, optim, scheduler, epoch, custom_path=args.output)
        print("successful fine-tuning...")
        print("Testing Model Training Results With Validation Prompt...")
        
        # Initialize an empty list to store the model generation strings
        generated_strings = []

        for x in range(10):
            ModelGeneration = infer_txt_generator("Albert Einstein was ", model, tokenizer)
            print(ModelGeneration)
            generated_strings.append(ModelGeneration)

        # Write the generated strings to a CSV file
        with open(output_file_path, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['Generated Strings'])
            for string in generated_strings:
                csv_writer.writerow([string])
        print("Model output written to CSV file:", output_file_path)

        all_inclusive_plots(args.output, 'graphs')
    except Exception as e:
        print("Error: ", e)
    finally:
        monitor.stop_monitoring()


# Have the model conduct Inferences
if __name__ == '__main__':
    main()
    