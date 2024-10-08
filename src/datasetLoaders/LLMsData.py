# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Validated on GPT2-Large With OpenWebText
# Backend: Pytorch
import os
from datasets import load_dataset
from transformers import GPTQConfig, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch

context_length = int(os.environ.get('MAX_TOK_LENGTH')) if os.environ.get('MAX_TOK_LENGTH') is not None else 128


# GENERAL CLASS FOR LOADING DATASETS FOR LLM
class HFDatasetLoader:
    """
    Parameters
    tokenizer : Tokenizer
        The tokenizer to use for tokenizing the text data.
    dataset_name : str
        The name of the dataset to load.
    split : str, optional
        The split of the dataset to load (default is 'train').
    context_length : int, optional
        The maximum length of the context (default is 512).

    Description:
        This class is a general dataset loader for loading datasets for large language models (LLMs). It
        loads the specified dataset with the specified split, and uses the provided tokenizer to tokenize the text data.
    """
    def __init__(self, tokenizer, dataset_name, split='train', context_length=512):
        # Load the specified dataset with the specified split
        self.data = load_dataset(dataset_name, split=split, trust_remote_code=True)
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=self.context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class WikiTextDataset: 
    """
    Parameters
    tokenizer : Tokenizer
        The tokenizer to use for tokenizing the text data.
    split : str
        The split of the dataset to load.

    Description:
        This class is a dataset loader for the WikiText dataset. It loads the WikiText dataset with the
        specified split, and uses the provided tokenizer to tokenize the text data.

    Link: https://huggingface.co/datasets/wikitext
    """
    def __init__(self, tokenizer, split):
        # Load the WikiText dataset with the specified split
        self.data = load_dataset('wikitext', 'wikitext-103-v1', split=split, trust_remote_code=True)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class YouTubeCommonsDataset:
    
    """
    Parameters
    tokenizer : Tokenizer
        The tokenizer to use for tokenizing the text data.
    split : str
        The split of the dataset to load.

    Description:
        This class is a dataset loader for the YouTube-Commons dataset. It loads the YouTube-Commons dataset
        with the specified split, and uses the provided tokenizer to tokenize the text data.

    Link: https://huggingface.co/datasets/PleIAs/YouTube-Commons
    """
    def __init__(self, tokenizer, split):
        # Load the YouTube-Commons dataset with the specified split
        self.data = load_dataset('PleIAs/YouTube-Commons', split=split)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class BookCorpusDataset:
    """
    Parameters
    tokenizer : Tokenizer
        The tokenizer to use for tokenizing the text data.
    split : str, optional
        The split of the dataset to load (default is 'train').

    Description:
        This class is a dataset loader for the BookCorpus dataset. It loads the BookCorpus dataset with the
        specified split, and uses the provided tokenizer to tokenize the text data.

    Link: https://huggingface.co/datasets/bookcorpus
    """
    def __init__(self, tokenizer, split='train'):
        # Load the BookCorpus dataset with the specified split
        self.data = load_dataset('bookcorpus', split=split, trust_remote_code=True)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class OpenWebTextDataset:
    
    """
    Parameters
    tokenizer : Tokenizer
        The tokenizer to use for tokenizing the text data.
    split : str, optional
        The split of the dataset to load (default is 'train').

    Description:
        This class is a dataset loader for the OpenWebText dataset. It loads the OpenWebText dataset with the
        specified split, and uses the provided tokenizer to tokenize the text data.
    
    Link: https://huggingface.co/datasets/Skylion007/openwebtext
    """
    def __init__(self, tokenizer, split='train'):
        # Load the OpenWebText dataset with the specified split
        self.data = load_dataset("openwebtext", split=split, trust_remote_code=True)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class FineWebDataset:
    
    """
    Parameters
    tokenizer : Tokenizer
        The tokenizer to use for tokenizing the text data.
    split : str, optional
        The split of the dataset to load (default is 'train').

    Description:
        This class is a dataset loader for the FineWeb dataset. It loads the FineWeb dataset with the
        specified split, and uses the provided tokenizer to tokenize the text data.
    
    Link: https://huggingface.co/datasets/HuggingFaceFW/fineweb
    """
    def __init__(self, tokenizer, split):
        # Load the FineWeb dataset with the specified split
        self.data = load_dataset('HuggingFaceFW/fineweb', split=split)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class FalconRefinedWebDataset:
    """
    Parameters
    tokenizer : Tokenizer
        The tokenizer to use for tokenizing the text data.
    split : str, optional
        The split of the dataset to load (default is 'train').

    Description:
        This class is a dataset loader for the Falcon-RefinedWeb dataset. It loads the Falcon-RefinedWeb dataset with the
        specified split, and uses the provided tokenizer to tokenize the text data.

    Link: https://huggingface.co/datasets/tiiuae/falcon-refinedweb
    """
    
    def __init__(self, tokenizer, split='train'):
        # Load the Falcon-RefinedWeb dataset with the specified split
        self.data = load_dataset('tiiuae/falcon-refinedweb', split=split)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class PileDataset:
    """
    Parameters
    tokenizer : Tokenizer
        The tokenizer to use for tokenizing the text data.
    split : str, optional
        The split of the dataset to load (default is 'train').

    Description:
        This class is a dataset loader for the Pile dataset. It loads the Pile dataset with the
        specified split, and uses the provided tokenizer to tokenize the text data.
    
    Link: https://huggingface.co/datasets/EleutherAI/pile
    """
    def __init__(self, tokenizer, split='train'):
        # Load the 'all' subset of the Pile dataset with the specified split
        self.data = load_dataset("EleutherAI/pile", "all", split=split, trust_remote_code=True)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class RedPajamaDataset:
    """
    Parameters
    tokenizer : Tokenizer
        The tokenizer to use for tokenizing the text data.
    split : str, optional
        The split of the dataset to load (default is 'train').

    Description:
        This class is a dataset loader for the RedPajama-Data-1T-Sample dataset. It loads the RedPajama-Data-1T-Sample dataset with the
        specified split, and uses the provided tokenizer to tokenize the text data.

    Link: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T
    """
    def __init__(self, tokenizer, split='train'):
        # Load the RedPajama-Data-1T-Sample dataset with the specified split
        self.data = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split, trust_remote_code=True)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class OscarDataset:
    """
    Parameters
    tokenizer : Tokenizer
        The tokenizer to use for tokenizing the text data.
    split : str, optional
        The split of the dataset to load (default is 'train').

    Description:
        This class is a dataset loader for the OSCAR dataset. It loads the OSCAR dataset with the
        specified split, and uses the provided tokenizer to tokenize the text data.

    Link: https://huggingface.co/datasets/oscar
    """
    def __init__(self, tokenizer, split='train'):
        # Load the 'unshuffled_deduplicated_en' subset of the OSCAR dataset with the specified split
        self.data = load_dataset("oscar", "unshuffled_deduplicated_en", split=split, trust_remote_code=True)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class StarCoderDataset:
    """
    Parameters
    tokenizer : Tokenizer
        The tokenizer to use for tokenizing the text data.
    data_dir : str, optional
        The data directory for the dataset (default is 'python').
    split : str, optional
        The split of the dataset to load (default is 'train').

    Description: 
        This class is a dataset loader for the StarCoder dataset. It loads the StarCoder dataset with the
        specified split and data directory, and uses the provided tokenizer to tokenize the text data.
    
    Link: https://huggingface.co/datasets/bigcode/starcoderdata
    """
    def __init__(self, tokenizer, data_dir='python', split='train'):
        # Load the StarCoder dataset with the specified split and data directory
        self.data = load_dataset("bigcode/starcoderdata", data_dir=data_dir, split=split, trust_remote_code=True)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


# Loaders
def open_wikiText(collate_fn=None, tokenizer=None, Ben_batch_size=None):
    """
    Parameters
    None

    Returns
    TrainingChatData : DataLoader
        DataLoader object containing the training data for the model.
    ValidatingChatData : DataLoader
        DataLoader object containing the validation data for the model.

    Description:
        This function loads and preprocesses the training and validation data files for a chatbot model using
        the WikiText dataset. It performs the following steps: Loads the training data files. Loads the validation data
        files. Preprocesses the data. Creates distributed versions of the datasets. Returns the DataLoader objects for
        training and validation data.
    """
    print('Loading Training Data Files...')
    train_data = WikiTextDataset(split='train', tokenizer=tokenizer)
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=Ben_batch_size, shuffle=True, collate_fn=collate_fn)
    return TrainingChatData

def open_WebText(collate_fn=None, tokenizer=None, Ben_batch_size=None):
    """
    Parameters
    None

    Returns
    TrainingChatData : DataLoader
        DataLoader object containing the training data for the model.
    ValidatingChatData : DataLoader
        DataLoader object containing the validation data for the model.

    Description:
        This function loads and preprocesses the training and validation data files for a chatbot model using
        the OpenWebText and WikiText datasets. It performs the following steps:
        Loads the training data files from OpenWebTextDataset.
        Loads the validation data files from WikiTextDataset.
        Preprocesses the data.
        Creates distributed versions of the datasets.
        Returns the DataLoader objects for training and validation data.
    """
    print('Loading Training Data Files...')
    train_data = OpenWebTextDataset(tokenizer, 'train')
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=Ben_batch_size, shuffle=True, collate_fn=collate_fn)
    return TrainingChatData


def open_RedPajama(collate_fn=None, tokenizer=None, Ben_batch_size=None):
    """
    Parameters
    None

    Returns
    TrainingChatData : DataLoader
        DataLoader object containing the training data for the model.

    Description: 
        This function loads and preprocesses the training data files for a chatbot model using
        the RedPajama-Data-1T-Sample dataset. It performs the following steps:
        Loads the training data files from RedPajamaDataset.
        Preprocesses the data.
        Creates distributed versions of the datasets.
        Returns the DataLoader objects for training data.
    """
    print('Loading Training Data Files...')
    train_data = RedPajamaDataset(tokenizer, 'train')
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=Ben_batch_size, shuffle=True, collate_fn=collate_fn)
    return TrainingChatData


def load_PileData(collate_fn=None, tokenizer=None, Ben_batch_size=None):
    """
    Parameters
    None

    Returns
    TrainingChatData : DataLoader
        DataLoader object containing the training data for the model.

    Description:
        This function loads and preprocesses the training data files for a chatbot model using
        the Pile dataset. It performs the following steps:
        Loads the training data files from PileDataset.
        Preprocesses the data.
        Creates distributed versions of the datasets.
        Returns the DataLoader objects for training data.
    """
    print('Loading Training Data Files...')
    train_data = PileDataset(tokenizer, 'train')
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=Ben_batch_size, shuffle=True, collate_fn=collate_fn)
    return TrainingChatData


def load_StarCoder(coding_language='python', collate_fn=None, tokenizer=None, Ben_batch_size=None):
    """
    Parameters
    coding_language : str
        specified dataset subset for https://huggingface.co/datasets/bigcode/starcoderdata

    Returns
    TrainingChatData : DataLoader
        DataLoader object containing the training data for the model.

    Description:
        This function loads and preprocesses the training data files for a chatbot model using
        the StarCoder dataset. It performs the following steps:
        Loads the training data files from StarCoderDataset.
        Preprocesses the data.
        Creates distributed versions of the datasets.
        Returns the DataLoader objects for training data.
    """
    print('Loading Training Data Files...')
    train_data = StarCoderDataset(tokenizer, data_dir=coding_language, split='train')
    print("Preprocessing...")
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=Ben_batch_size, shuffle=True, collate_fn=collate_fn)
    return TrainingChatData

# Added to enable easy Quantization of popular models and discrete loading of different precisions
def load_HFModel(model_name = None, tokenizer = None, quantized = False, bit_precision = 4, ):
    """
    Parameters
    model_name : str
        The name of the model to load.
    tokenizer : Tokenizer, optional
        The tokenizer to use for tokenizing the text data.
    quantized : bool, optional
        Whether to load the quantized model (default is False).
    precision : str, optional
        The precision of the model to load (default is 'fp32').

    Returns
    model : Model
        The model to load.
    """
    # Check if tokenizer is None, if so load the tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        tokenizer = tokenizer

    # processing for Quantization argument
    if quantized == True:
        Q_config = GPTQConfig(bits = bit_precision, tokenizer=tokenizer,)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=Q_config, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    return model
