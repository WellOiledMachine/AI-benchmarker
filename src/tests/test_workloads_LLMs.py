# NEEDS UPDATED: NO LONGER WORKS - Cody Sloan, 7/25/24

import unittest
import sys
from unittest.mock import Mock
import torch
from workloads_LLMs import infer_txt_generator, train_model_txt_generator, pretrain_model_txt_generator, Grab_Tokenizer_Custom, Grab_Tokenizer, collate_fn_txt_generator, collate_fn_BERT

# sys.path.append('./hpc-profiler/hpc-profiler')  # Use the path to the directory of your Python file


class TestInferTxtGenerator(unittest.TestCase):
    def test_infer_txt_generator(self):
        # Mock the model and tokenizer
        model = Mock()
        tokenizer = Mock()

        # Set the return values for the model and tokenizer
        model.generate.return_value = [1, 2, 3]
        tokenizer.eos_token_id = 3

        # Call the function with a prompt
        generated_text = infer_txt_generator("Hello, how are you?", model, tokenizer)

        # Check the output
        self.assertEqual(generated_text, "Hello, how are you?")

class TestTrainModelTxtGenerator(unittest.TestCase):
    def setUp(self):
        # Mock the DataLoader, model, accelerator, optimizer, and scheduler
        self.trainer_data = Mock()
        self.model = Mock()
        self.accelerator = Mock()
        self.optimizer = Mock()
        self.scheduler = Mock()

        # Configure the DataLoader to return some mock data
        self.mock_batch = (torch.tensor([[1, 2, 3]]), torch.tensor([1]))
        self.trainer_data.__iter__.return_value = iter([self.mock_batch, None])  # Include a None to simulate end of data

        # Mock the model's output
        self.model.return_value = Mock(logits=Mock(size=Mock(return_value=(10, 10))))

    @patch('hpc-profiler.workloads_LLMs.tqdm')
    @patch('hpc-profiler.workloads_LLMs.open', new_callable=unittest.mock.mock_open)
    @patch('hpc-profiler.workloads_LLMs.csv.writer')
    @patch('hpc-profiler.workloads_LLMs.psutil.disk_io_counters')
    def test_train_model_txt_generator(self, mock_disk_io_counters, mock_csv_writer, mock_open, mock_tqdm):
        # Set up the disk IO counters
        mock_disk_io_counters.return_value.read_count = 100
        mock_disk_io_counters.return_value.write_count = 50

        # Call the function
        train_model_txt_generator(self.trainer_data, self.model, self.accelerator, self.optimizer, self.scheduler, epoch_pass=1)

        # Check that the model's train method was called
        self.model.train.assert_called_once()

        # Check that the optimizer's zero_grad method was called
        self.optimizer.zero_grad.assert_called()

class TestPretrainModelTxtGenerator(unittest.TestCase):
    def setUp(self):
        # Mock the DataLoader, model, accelerator, optimizer, and scheduler
        self.training_data_loader = Mock()
        self.model = Mock()
        self.accelerator = Mock()
        self.optimizer = Mock()
        self.scheduler = Mock()

        # Configure the DataLoader to return some mock data
        self.mock_batch = ([Mock()], [Mock()])
        self.training_data_loader.__iter__.return_value = iter([self.mock_batch, None])  # Include a None to simulate end of data

        # Mock the model's output
        self.model.return_value = Mock(logits=Mock(size=Mock(return_value=(10, 10))))

    @patch('hpc-profiler.workloads_LLMs.tqdm')
    @patch('hpc-profiler.workloads_LLMs.open', new_callable=unittest.mock.mock_open)
    @patch('hpc-profiler.workloads_LLMs.csv.writer')
    @patch('hpc-profiler.workloads_LLMs.psutil.disk_io_counters')
    def test_pretrain_model_txt_generator(self, mock_disk_io_counters, mock_csv_writer, mock_open, mock_tqdm):
        # Set up the disk IO counters
        mock_disk_io_counters.return_value.read_count = 100
        mock_disk_io_counters.return_value.write_count = 50

        # Call the function
        pretrain_model_txt_generator(self.training_data_loader, self.model, self.accelerator, self.optimizer, self.scheduler, total_epochs=1)

        # Check that the model's train method was called
        self.model.train.assert_called_once()

        # Check that the optimizer's zero_grad method was called
        self.optimizer.zero_grad.assert_called()
        # Check that the model's forward method was called
        self.model.assert_called()

        # Check that the optimizer's step method was called
        self.optimizer.step.assert_called()

        # Check that the scheduler's step method was called
        self.scheduler.step.assert_called()

        # Check that the accelerator's backward method was called
        self.accelerator.backward.assert_called()

        # Check that the accelerator's free_memory method was called
        self.accelerator.free_memory.assert_called()

        # Check that file operations were performed
        mock_open.assert_called_with('training_results.csv', mode='w', newline='')
        mock_csv_writer.assert_called()

class TestGrabTokenizerCustom(unittest.TestCase):
    @patch('hpc-profiler.workloads_LLMs.AutoTokenizer')
    def test_grab_tokenizer_custom(self, mock_auto_tokenizer):
        # Setup the mock for AutoTokenizer.from_pretrained
        mock_tokenizer_instance = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Call the function
        tokenizer_type = 'gpt2-large'
        pad_token = '<pad>'
        eos_token = '<eos>'
        bos_token = '<bos>'
        tokenizer = Grab_Tokenizer_Custom(tokenizer_type, pad_token, eos_token, bos_token)

        # Check that from_pretrained was called correctly
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(tokenizer_type, use_fast=True)

        # Check that the tokens were set correctly
        self.assertEqual(tokenizer.pad_token, pad_token)
        self.assertEqual(tokenizer.eos_token, eos_token)
        self.assertEqual(tokenizer.bos_token, bos_token)

        # Check that the returned object is the mock instance
        self.assertEqual(tokenizer, mock_tokenizer_instance)

class TestGrabTokenizer(unittest.TestCase):
    @patch('hpc-profiler.workloads_LLMs.AutoTokenizer')
    def test_grab_tokenizer(self, mock_auto_tokenizer):
        # Setup the mock for AutoTokenizer.from_pretrained
        mock_tokenizer_instance = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Call the function
        tokenizer_type = 'gpt2-large'
        tokenizer = Grab_Tokenizer(tokenizer_type)

        # Check that from_pretrained was called correctly
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(tokenizer_type)

        # Check that the returned object is the mock instance
        self.assertEqual(tokenizer, mock_tokenizer_instance)

class TestCollateFnTxtGenerator(unittest.TestCase):
    def test_collate_fn_txt_generator(self):
        # Create mock data for the batch
        batch = [
            {'input_ids': torch.tensor([1, 2, 3]), 'attention_mask': torch.tensor([1, 1, 1])},
            {'input_ids': torch.tensor([1, 2]), 'attention_mask': torch.tensor([1, 1])}
        ]

        # Expected padded results
        expected_input_ids = torch.pad_sequence([b['input_ids'] for b in batch], batch_first=True)
        expected_attention_masks = torch.pad_sequence([b['attention_mask'] for b in batch], batch_first=True)

        # Call the function
        input_ids, attention_masks = collate_fn_txt_generator(batch)

        # Check the results
        self.assertTrue(torch.equal(input_ids, expected_input_ids))
        self.assertTrue(torch.equal(attention_masks, expected_attention_masks))

class TestCollateFnBERT(unittest.TestCase):
    def test_collate_fn_bert(self):
        # Create mock data for the batch
        batch = [
            {'input_ids': torch.tensor([1, 2, 3]), 'attention_mask': torch.tensor([1, 1, 1]), 'token_type_ids': torch.tensor([0, 0, 0])},
            {'input_ids': torch.tensor([1, 2]), 'attention_mask': torch.tensor([1, 1]), 'token_type_ids': torch.tensor([0, 0])}
        ]

        # Expected padded results
        expected_input_ids = torch.pad_sequence([b['input_ids'] for b in batch], batch_first=True)
        expected_attention_masks = torch.pad_sequence([b['attention_mask'] for b in batch], batch_first=True)
        expected_token_type_ids = torch.pad_sequence([b['token_type_ids'] for b in batch], batch_first=True)

        # Call the function
        input_ids, attention_masks, token_type_ids = collate_fn_BERT(batch)

        # Check the results
        self.assertTrue(torch.equal(input_ids, expected_input_ids))
        self.assertTrue(torch.equal(attention_masks, expected_attention_masks))
        self.assertTrue(torch.equal(token_type_ids, expected_token_type_ids))