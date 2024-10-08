from datasets import load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import os
import pickle
import unicodedata
from collections import defaultdict
import matplotlib.pyplot as plt
from datasets import Audio, load_dataset, load_from_disk
import torchaudio
from speechbrain.inference import EncoderClassifier
import psutil
import numpy as np
from tqdm import tqdm
from datasets import Array2D

# Helper Function for Data Visualization
def print_dict_structure(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key) + ':', end=" ")
        if isinstance(value, dict):
            print()  # Move to the next line before printing nested dictionary
            print_dict_structure(value, indent+1)
        elif isinstance(value, list):
            print(f"List of {len(value)} items")
        else:
            print(str(value))
# Preprocess the dataset for TTS training, There are various preprocessing steps that can be added here.
# This includes speaker embedding extraction, text normalization, and filtering out long sequences.
# The dataset is cached to disk to speed up subsequent runs should we want to profile multiple models on the same dataset.
# This doesn't have multilingual support yet and just assumes and english dataset for initial testing.
# This will later be Generalized.
@dataclass
class TTSDatasetProcessor:
    """
    Text-to-Speech (TTS) Dataset Processor

    This class is designed to handle the preprocessing of datasets for TTS model training. 
    It includes methods for normalizing text, processing audio data, extracting speaker embeddings, 
    and preparing the dataset for training or evaluation.

    Attributes
    ----------
    dataset_name : str
        The name of the dataset to be processed.
    processor : Any
        The processor used for audio and text processing.
    split : str, optional
        The specific split of the dataset to be used (default is 'train').
    sampling_rate : int, optional
        The sampling rate of the audio data (default is 16000 Hz).
    custom_cache_path : str, optional
        Custom path for caching processed data (default is an empty string).
    config : Any, optional
        Configuration for the dataset or model (default is None).
    speaker_model_name : str
        The name of the model used for speaker embedding extraction.
    speaker_model : EncoderClassifier
        The loaded speaker model for embedding extraction.
    embedding_size : int
        The size of the extracted speaker embeddings.

    Methods
    -------
    normalize_text(text)
        Normalize text by removing accent marks from characters.
    prepare_dataset(example)
        Prepare a single dataset entry by normalizing text and processing audio.
    load_and_extract_embeddings(audio_path)
        Load an audio file and extract normalized speaker embeddings.
    is_not_too_long(example)
        Check if the sequence length of an example is within acceptable limits.
    process_dataset()
        Main method to process the entire dataset, including loading, preprocessing, and caching.
    is_speaker_count_valid(example, speaker_counts)
        Validate if the speaker count for a given example is within the valid range.
    get_project_directory()
        Retrieve the absolute path of the project directory.

    Notes
    -----
    - This class is crucial for ensuring that the dataset is properly prepared for TTS model training, which includes handling of audio data and text normalization.
    - The class methods are designed to be used sequentially to process and prepare the dataset.
    - Proper configuration and initialization of the class attributes are essential for the correct functioning of the methods.
    """
    def __init__(self, dataset_name, processor, split='train', sampling_rate=16000, custom_cache_path='', config=None, cpu_count=None, load_from_disk=False, load_from_cache=False):
        self.dataset_name = dataset_name
        self.processor = processor
        self.split = split
        self.sampling_rate = sampling_rate
        self.custom_cache_path = custom_cache_path
        self.speaker_model_name: str = 'speechbrain/spkrec-xvect-voxceleb'
        self.config = config  # Default model
        self.speaker_model = EncoderClassifier.from_hparams(source=self.speaker_model_name, run_opts={"device": "cuda"}, savedir=f"{self.custom_cache_path}/cached_models/speaker_embeddings")
        self.embedding_size = 512  # Set this based on your model
        self.cpu_count = cpu_count if cpu_count else int(psutil.cpu_count() / 2)
        self.load_from_cache = load_from_cache
        self.load_from_disk = load_from_disk

    def normalize_text(self, text):
        """
        Normalizes the input text by removing diacritical marks (accents) from characters.

        This function uses Unicode Normalization Form D (NFD) to decompose characters into their
        composite parts. It then filters out the diacritical marks, which are categorized under
        the Unicode category 'Mn' (Mark, Nonspacing), and reconstructs the text without these marks.

        Parameters:
            text (str): The text string to be normalized.

        Returns:
            str: The normalized text with diacritical marks removed.

        Examples:
            >>> normalize_text("Café Münster")
        'Cafe Munster'

        Notes:
            - This method is particularly useful for preprocessing text in natural language processing
            tasks where diacritical marks may not be relevant.
            - The method does not modify other aspects of the text, such as casing or punctuation.
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    # def prepare_multispeaker_dataset_facebook(self, example):
    #     """
    #     Prepare dataset by normalizing text and processing audio.
        
    #     Args:
    #         example (Dict[str, Any]): A dictionary representing a single dataset entry. It must contain the following keys:
    #         audio (Dict[str, Any]): A sub-dictionary that includes the audio data and metadata. It should have at least the following sub-keys:
    #         path (str): The full file path to the audio file.
    #         array (np.ndarray): The audio data array.
    #         sampling_rate (int): The sampling rate of the audio data.
    #         speaker_id (str or int): The identifier for the speaker in the audio file.
    #         chapter_id (str or int): The identifier for the chapter from which the audio is taken.
    #         text (str): The transcript or text associated with the audio file.
    #     Returns:
    #         processed_example (Dict[str, Any]): A dictionary containing the processed data ready for use in TTS model training or evaluation. It includes:
    #         text (str): The normalized text from the input example.
    #         audio_target (np.ndarray): The processed audio data array.
    #         sampling_rate (int): The sampling rate of the audio data.
    #         speaker_embeddings (np.ndarray): The extracted speaker embeddings as a NumPy array.
    #     """
    #     audio = example["audio"]
    #     # Extract speaker ID and chapter ID from the dataset entry
    #     speaker_id = example["speaker_id"]
    #     chapter_id = example["chapter_id"]
    #     audio_p = example["audio"]["path"]  # Correct variable name used here

    #     # Extract the file name from the path
    #     audio_file = os.path.basename(audio_p)  # Use os.path.basename to extract the file name

    #     # Extract the directory path without the file name
    #     directory_path = os.path.dirname(audio_p)

    #     # Construct the path based on the dataset structure
    #     # The default path stored in the sample is fucked. This is bullshit, poorly named and needs to be fixed. - Tyson Limato
    #     # Assuming the directory_path already includes the necessary path up to the dataset specific folder
    #     audio_path = os.path.join(directory_path, "LibriSpeech", "train-other-500", str(speaker_id), str(chapter_id), audio_file)

    #     #print("File exists:", os.path.exists(audio_path))
    #     #print("File is accessible:", os.access(audio_path, os.R_OK))

    #     # Now use audio_path to load and process the audio
    #     processed_example = self.processor(
    #         text=example["text"],
    #         audio_target=audio["array"],
    #         sampling_rate=audio["sampling_rate"],
    #         return_attention_mask=False,
    #     )
    #     # Assuming you have a method `extract_speaker_embeddings` that takes the audio array and sampling rate
    #     # and returns the speaker embeddings.
    #     processed_example["speaker_embeddings"] = self.load_and_extract_embeddings(audio_path=audio_path)

    #     return processed_example
    
    def prepare_multispeaker_dataset_alt(self, example):
        """
        Prepare dataset by processing audio.
        
        Args:
            example (Dict[str, Any]): A dictionary representing a single dataset entry. It must contain the following keys:
            audio (Dict[str, Any]): A sub-dictionary that includes the audio data and metadata. It should have at least the following sub-keys:
            path (str): The full file path to the audio file.
            array (np.ndarray): The audio data array.
            sampling_rate (int): The sampling rate of the audio data.
        Returns:
            processed_example (Dict[str, Any]): A dictionary containing the processed data ready for use in TTS model training or evaluation. It includes:
            audio_target (np.ndarray): The processed audio data array.
            sampling_rate (int): The sampling rate of the audio data.
        """
        audio = example["audio"]
        audio_path = audio["path"]  # Use the full path directly from the audio dictionary
        two_levels_up = os.path.abspath(os.path.join(audio_path, '..', '..'))
        one_level_up = os.path.abspath(os.path.join(audio_path, '..'))
        speaker_id = os.path.basename(two_levels_up)
        print(f"Speaker ID: {speaker_id}")
        chapter_id = os.path.basename(one_level_up)
        print(f"Chapter ID: {chapter_id}")

        if not audio_path or not os.path.exists(audio_path):
            raise ValueError(f"Invalid or missing audio path: {audio_path}")
        
        # Get transcription from text file
        transcription = self.get_transcription_from_file(audio_path, speaker_id=speaker_id, chapter_id=chapter_id)
        if not transcription:
            raise ValueError(f"Transcription not found for audio: {audio_path}")
        
        print(f"Audio path: {audio_path}")
        print(f"Transcription: {transcription}")

        # Tokenize the text
        input_ids = self.processor(text=transcription).input_ids
        example["input_ids"] = input_ids

        # Now use audio_path to load and process the audio
        example = self.processor(
            audio_target=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=False,
        )
        # strip off the batch dimension
        # Assuming you have a method `extract_speaker_embeddings` that takes the audio array and sampling rate
        # and returns the speaker embeddings.
        # Note: Adjust this part if you do not need speaker embeddings or if the method to extract them is different.
        # strip off the batch dimension
        example["labels"] = example["input_values"]

        example["speaker_embeddings"] = self.create_speaker_embedding(audio["array"])

        print(list(example.keys()))

        return example
    
    def get_transcription_from_file(self, audio_path, speaker_id, chapter_id):
        """
        Extract transcription from the corresponding text file.
        
        Args:
            audio_path (str): Path to the audio file.
        
        Returns:
            str: The transcription for the given audio file.
        """
        directory = os.path.dirname(audio_path)
        filename = os.path.basename(audio_path)
        base_filename = os.path.splitext(filename)[0]
        
        # Assuming the transcription file is in the same directory and has a .txt extension
        trans_file = os.path.join(directory, f"{speaker_id}-{chapter_id}.trans.txt")
        
        if not os.path.exists(trans_file):
            print(f"Transcription file not found: {trans_file}")
            return None
        
        with open(trans_file, "r") as f:
            for line in f:
                if line.startswith(base_filename):
                    return line[len(base_filename):].strip()
        
        print(f"Transcription not found for {filename}")
        return None
    
    def create_speaker_embedding(self, waveform):
        with torch.no_grad():
            speaker_embeddings = self.speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
        return speaker_embeddings

    def load_and_extract_embeddings(self, audio_path):
        """Load an audio file and extract normalized speaker embeddings."""
        signal, fs = torchaudio.load(audio_path, format='flac',)

        # Check if the sampling rate is correct
        assert fs == self.sampling_rate, f"Expected {self.sampling_rate}Hz, got {fs}Hz"

        with torch.no_grad():
            embeddings = self.speaker_model.encode_batch(signal.to('cuda'))
            embeddings = torch.nn.functional.normalize(embeddings, dim=2)
        return embeddings.squeeze().cpu().numpy()

    def is_not_too_long(self, example):
        """Check if the sequence of input_ids is not too long."""
        max_length = 200  # Define your maximum length threshold here
        return len(example["input_ids"]) < max_length

    def process_multispeaker_dataset(self):
        """
        Processes the TTS dataset for multispeaker scenarios by loading data, filtering,
        and preparing it for training. This includes downloading the dataset if not cached,
        casting audio data to the specified sampling rate, filtering out speakers with too few examples,
        and mapping the dataset through a processing function that extracts speaker embeddings.

        The method handles caching to avoid reprocessing data unnecessarily. If a cached version of the
        dataset is available, it loads from there; otherwise, it processes and then caches the data.

        Attributes:
            project_directory (str): The directory where the dataset and cache will be stored.
                                    This is determined by `self.custom_cache_path` if set, otherwise by `self.get_project_directory()`.
            cache_path (str): The path to the cache file where the processed dataset is stored.
            train_dataset (Dataset): The dataset loaded from Hugging Face or from cache, processed for training.

        Returns:
            Dataset: A `datasets.Dataset` object that has been processed and is ready for training use.
                    This dataset includes only the training split, with audio data cast to the specified
                    sampling rate and filtered to include only valid speaker examples.

        Raises:
            FileNotFoundError: If the dataset cannot be loaded from Hugging Face and no cache is available.
            Exception: General exceptions related to file handling or dataset processing could be raised.

        Notes:
            - The method uses `psutil` for CPU count when setting `num_proc` in dataset mapping, which should be considered when running on different environments.
            - Significant memory issues can occur if the dataset is large; hence, careful handling of batch sizes and multiprocessing is advised.
        """

        # Cache Handling / Preparations
        project_directory = self.custom_cache_path if self.custom_cache_path != '' else self.get_project_directory()
        cache_path = f"{project_directory}/cached_data/{self.split}_processed_dataset.pkl"
        os.makedirs(f"{project_directory}/cached_data", exist_ok=True)

        if os.path.exists(cache_path) and self.load_from_cache:
            print(f"Loading cached data from {cache_path}")
            try:
                with open(cache_path, 'rb') as cache_file:
                    dataset = pickle.load(cache_file)
                    return dataset['train']
            except Exception as e:
                print(f"Failed to load cached data: {e}")
                print("Proceeding to download and process the dataset...")

        print(f"Downloading and/or loading {self.dataset_name} dataset from Hugging Face...")
        if isinstance(self.dataset_name, str) and not self.load_from_disk:
            train_dataset = load_dataset(self.dataset_name, self.config, split='train', trust_remote_code=True)
        else:
            train_dataset = load_dataset(self.dataset_name, split='train')

        print("Printing the first entry of the train dataset")
        print_dict_structure(train_dataset[0])  # Adjust the number to print more or fewer entries

        train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        print("Printing the first entry of the train dataset after casting")
        print_dict_structure(train_dataset[0])  # Adjust the number to print more or fewer entries\

        print("Grabbing only first 1000 for DEBUGGING")
        train_dataset = train_dataset.select(range(1000))
        print_dict_structure(train_dataset[0])  # Adjust the number to print more or fewer entries
        train_dataset = train_dataset.map(
            self.prepare_multispeaker_dataset_alt,
            remove_columns=train_dataset.column_names,
            desc="Processing dataset",
            num_proc=self.cpu_count,
            load_from_cache_file=False
        )
        print("Printing the first entry of the train dataset after processing")
        print_dict_structure(train_dataset[0])

        print("Showing Spectrogram of the first entry of the train dataset")
        test_example = train_dataset[0]
        test_example = np.array(test_example["labels"]).T
        print(test_example.shape)

        plt.figure()
        plt.imshow(test_example, cmap='viridis', aspect='auto')
        plt.show()

        print(train_dataset[0])

        try:
            with open(cache_path, 'wb') as cache_file:
                pickle.dump({'train': train_dataset}, cache_file)
                print(f"Processed samples cached to {cache_path}")
        except Exception as e:
            print(f"Failed to cache processed data: {e}")
        
        return train_dataset

    def get_project_directory(self):
        """Get the absolute path of the project directory."""
        current_dir = os.path.dirname(os.path.realpath(__file__))
        project_directory = os.path.abspath(os.path.join(current_dir, '..', '..'))
        return project_directory


    def is_label_count_valid(self, example, label_counts, min_count=20, max_count=100000000):
        """Check if the label count for the given example is within the valid range."""
        label = example['label']
        return min_count <= label_counts[label] <= max_count

    @staticmethod
    def load_tts_dataset(dataset_name="facebook/voxpopuli", split=None, lang=None, config=None, cache_dir=None):
        """
        Load a TTS dataset.

        Args:
            dataset_name (str): The name or path of the dataset repository on Hugging Face.
            split (str, optional): The specific split of the dataset to load (e.g., 'train', 'test'). If None, loads the full dataset.
            cache_dir (str, optional): Directory where the datasets should be cached.

        Returns:
            Dataset or DatasetDict: The loaded dataset split if specified, otherwise the full dataset.
        """
        # Load the dataset with optional split and caching
        if split:
            if lang:
                if config:
                    dataset = load_dataset(dataset_name, lang, split=split, config=config, cache_dir=cache_dir, trust_remote_code=True)
                else:
                    dataset = load_dataset(dataset_name, lang, split=split, cache_dir=cache_dir, trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir, trust_remote_code=True)
        else:
            if lang:
                if config:
                    dataset = load_dataset(dataset_name, lang, config=config, cache_dir=cache_dir, trust_remote_code=True)
                else:
                    dataset = load_dataset(dataset_name, lang, cache_dir=cache_dir, trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_name, cache_dir=cache_dir, trust_remote_code=True)
            return dataset
    
    @staticmethod
    def get_project_directory():
        """
        Retrieves the absolute path of the project directory.

        This method computes the absolute path by determining the directory of the current script
        and navigating up two levels in the directory structure. This is typically used to locate
        the root directory of the project, especially in scenarios where relative paths from the
        script to other project resources need to be consistently resolved.

        Returns:
            str: The absolute path of the project directory.

        Examples:
            >>> TTSData.get_project_directory()
            '/path/to/project_directory'

        Notes:
            - This method assumes that the script is located two levels deep from the project root.
            If the directory structure changes, the method needs to be updated accordingly.
            - The method uses '__file__', which is the pathname of the file from which the module
            was loaded, if it was loaded from a file. The path is absolute unless the module was
                loaded from a relative path.
        """
        # Get the absolute path of the script being executed
        current_dir = os.path.dirname(os.path.realpath(__file__))
        # Navigate up two levels to get outside of 'hpc-profiler'
        project_directory = os.path.abspath(os.path.join(current_dir, '..', '..'))
        return project_directory

# @dataclass
# class TTSDataCollatorWithPadding:
#     processor: Any  # This should be an instance capable of handling padding of audio data
#     model: Any  # This should be your model configuration that might have specific needs like reduction factor

#     def __call__(self, features: List[Dict[str, Union[np.ndarray, int, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         # Extract audio data and potentially labels from features
#         audio_features = [feature['audio_target'] for feature in features if 'audio_target' in feature]
#         labels = [feature['label'] for feature in features if 'label' in feature]
#         speaker_embeddings = [feature['speaker_embeddings'] for feature in features if 'speaker_embeddings' in feature]

#         # Pad audio features using the processor's padding method
#         audio_padded = self.processor.pad(audio_features, return_tensors="pt")

#         # Create a batch dictionary to hold processed tensors
#         batch = {
#             'audio': audio_padded,
#         }

#         # If labels are present, handle them similarly
#         if labels:
#             labels_padded = self.processor.pad(labels, return_tensors="pt")
#             batch['labels'] = labels_padded

#         # Handle speaker embeddings if present
#         if speaker_embeddings:
#             # Assuming speaker embeddings are already tensors
#             speaker_embeddings_tensor = torch.stack(speaker_embeddings)
#             batch['speaker_embeddings'] = speaker_embeddings_tensor

#         # Additional handling for model-specific configurations
#         if hasattr(self.model, 'config') and hasattr(self.model.config, 'reduction_factor') and self.model.config.reduction_factor > 1:
#             target_lengths = torch.tensor([len(feature) for feature in audio_features])
#             target_lengths = target_lengths - (target_lengths % self.model.config.reduction_factor)
#             max_length = target_lengths.max()
#             batch['audio'] = batch['audio'][:, :max_length]

#         return batch
    
@dataclass
class TTSDataCollatorWithPadding:
    processor: Any
    model: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Ensure features are dictionaries
        if not all(isinstance(feature, dict) for feature in features):
            raise ValueError("Each feature should be a dictionary.")
        
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"][0]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # Pad the inputs and labels
        batch = self.processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # Replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # Remove decoder_attention_mask as it's not used during fine-tuning
        del batch["decoder_attention_mask"]
        del batch["attention_mask"]
        # Adjust target lengths for reduction factor if applicable
        if self.model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % self.model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # Add speaker embeddings to the batch
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        # Ensure labels have the correct shape
        if batch["labels"].dim() > 2:
            batch["labels"] = batch["labels"].view(batch["labels"].size(0), -1)

        return batch


# https://huggingface.co/learn/audio-course/chapter6/fine-tuning