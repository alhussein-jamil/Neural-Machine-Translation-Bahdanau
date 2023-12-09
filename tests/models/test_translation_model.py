import unittest
import torch
from torch import nn, optim
from models.translation_model import AlignAndTranslate
from models.encoder import Encoder
from models.decoder import Decoder 
from models.rnn import RNN


# HAVE TO VERIFY DIMENSIONS OF WEIGHTS OF FCNN 
class TestAlignAndTranslate(unittest.TestCase):

    def setUp(self):
        # Define parameters for testing
        encoder_params = {
            "rnn_hidden_size": 5,
            "rnn_num_layers": 1,
            "rnn_device": "cpu",
            "vocab_size": 5,
            "rnn_type": "GRU"
        }

        decoder_params = {
            "alignment": {
                "input_size": encoder_params["rnn_hidden_size"] * 2,
                "hidden_sizes": [10],
                "output_size": 5,
                "device": "cpu"
            },
            "rnn": {
                "input_size": encoder_params["rnn_hidden_size"] * 2,
                "hidden_size": 10,
                "num_layers": 1,
                "device": "cpu",
                "dropout": 0,
                "type": "GRU",
                "bidirectional": False,
            }
        }

        optimizer_params = {
            "lr": 0.001,  # Taux d'apprentissage  
            "momentum": 0.9,  # Moment 
            "weight_decay": 0,  # Terme de dégradation du poids 
        }

        criterion = torch.nn.CrossEntropyLoss()

        # Create an instance of the AlignAndTranslate model
        self.model = AlignAndTranslate(
            encoder_params=encoder_params,
            decoder_params=decoder_params,
            optimizer_params=optimizer_params,
            criterion=criterion
        )

    def test_model_initialization(self):
        # Check if the model is an instance of nn.Module
        self.assertIsInstance(self.model, nn.Module)

        # Check if the encoder and decoder are instances of their respective classes
        self.assertIsInstance(self.model.encoder, Encoder)
        self.assertIsInstance(self.model.decoder, Decoder)

        # Check if the optimizer and criterion are initialized correctly
        self.assertIsInstance(self.model.optimizer, optim.SGD)  
        self.assertIsInstance(self.model.criterion, nn.CrossEntropyLoss)

    def test_forward_pass(self):
        # Check if the forward pass runs without errors
        input_data = torch.randint(0, self.model.encoder.vocab_size, (32, 10)).to("cpu")
        output_tensor = self.model(input_data)

        # Adjust the expected shape based on your implementation
        expected_shape = torch.Size([32, 10, 5])  # Adapt based on your output size
        self.assertEqual(output_tensor.shape, expected_shape)

    def test_training_step(self):
        # Check if the training step runs without errors
        batch = {
            "english": {"idx": torch.randint(0, self.model.encoder.vocab_size, (32, 10), dtype=torch.long)},
            "french": {"idx": torch.randint(0, 5, (32, 10), dtype=torch.long)}
        }
        loss = self.model.train_step(batch["english"]["idx"], batch["french"]["idx"])

        # Ensure that the loss is a non-negative scalar value
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    def test_predict(self):
        # Check if the prediction runs without errors
        batch = {
            "english": {"idx": torch.randn(32, 10, self.model.encoder.vocab_size)},
            "french": {"idx": torch.randint(0, 5, (32, 10), dtype=torch.long)}
        }
        predictions = self.model.predict(batch["english"]["idx"])

        # Adjust the expected shape based on your implementation
        expected_shape = torch.Size([32, 10])  # Adapt based on your output size
        self.assertEqual(predictions.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()