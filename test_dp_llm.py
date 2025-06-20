import unittest
from unittest import mock

import dp_llm

class TrainDpLlmTest(unittest.TestCase):
    @mock.patch('dp_llm.PrivacyEngine')
    @mock.patch('dp_llm.DataLoader')
    @mock.patch('dp_llm.load_dataset')
    @mock.patch('dp_llm.AutoTokenizer')
    @mock.patch('dp_llm.AutoModelForCausalLM')
    def test_train_dp_llm_parameters(self, mock_model, mock_tokenizer, mock_load_dataset, mock_dataloader, mock_privacy_engine):
        mock_model.from_pretrained.return_value = mock.Mock(**{"to.return_value": mock.Mock(), "train.return_value": None})
        mock_tokenizer.from_pretrained.return_value = mock.Mock()
        dataset_mock = {'train': mock.Mock()}
        dataset_mock['train'].map.return_value = mock.Mock(set_format=mock.Mock(return_value=None))
        mock_load_dataset.return_value = dataset_mock
        mock_dataloader.return_value = []
        pe_instance = mock_privacy_engine.return_value
        pe_instance.make_private_with_epsilon.return_value = (mock_model.from_pretrained.return_value, mock.Mock(), [])

        out = dp_llm.train_dp_llm('gpt2', 'data.txt', 1.0, 1e-5, 1.0, output_dir='files')

        mock_model.from_pretrained.assert_called_with('gpt2')
        mock_tokenizer.from_pretrained.assert_called_with('gpt2')
        mock_load_dataset.assert_called_with('text', data_files={'train': 'data.txt'})
        pe_instance.make_private_with_epsilon.assert_called()
        self.assertTrue(out.endswith('dp_gpt2'))

if __name__ == '__main__':
    unittest.main()
