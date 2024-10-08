from workloads.TTSW import load_processor, train_tts_model
from datasetLoaders.TTSData import TTSDatasetProcessor
import torch
from accelerate import Accelerator
from transformers import AutoModelForTextToSpectrogram, SpeechT5ForTextToSpeech
from Profiler import monitor_system_utilization

if __name__ == "__main__":
    # Step 1: Load the processor
    model_name = "microsoft/speecht5_tts"
    """
    @inproceedings{panayotov2015librispeech,
        title={Librispeech: an ASR corpus based on public domain audio books},
        author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
        booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
        pages={5206--5210},
        year={2015},
        organization={IEEE}
    }

    """
    processor = load_processor("microsoft/speecht5_tts")
    accelerator = Accelerator()

    dataset_name = "librispeech_asr"
    data_monitor = monitor_system_utilization(benchmark_time_interval=1, model_name=dataset_name, custom_path="C:/Users/tylim/Documents/Contract Work/hpc-library/hpc-profiler/speecht5-tts/statistics")
    
    #Step 2: Load the dataset and process it
    print("Starting Data Processing")
    data_monitor.start_monitoring()
    # https://huggingface.co/datasets/openslr/librispeech_asr
    dataset_processor = TTSDatasetProcessor("C:/Users/tylim/.cache/huggingface/datasets/librispeech_asr/train-other-500", processor, load_from_cache=True, cpu_count=9, custom_cache_path="C:/Users/tylim/Documents/Contract Work/hpc-library/hpc-profiler/speecht5-tts")
    train_dataset = dataset_processor.process_multispeaker_dataset()
    data_monitor.stop_monitoring()
    print("Finished Processing Data")
    # Assuming we have a function to load the TTS model, optimizer, and scheduler
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    model.config.use_cache = False
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    
    # Step 3: Train the TTS model  # Assuming the dataset is split into train and test
    epochs = 3
    print("Starting Training")
    train_monitor = monitor_system_utilization(benchmark_time_interval=1, model_name=model_name, custom_path="C:/Users/tylim/Documents/Contract Work/hpc-library/hpc-profiler/speecht5-tts")
    train_monitor.start_monitoring()
    train_tts_model(model=model, model_name=model_name, processor=processor, train_data=train_dataset, optimizer=optimizer, epochs=epochs, custom_path="C:/Users/tylim/Documents/Contract Work/hpc-library/hpc-profiler/speecht5-tts")
    train_monitor.stop_monitoring()
    print("Finished Training")