import unittest

import torch
from torch import nn

from models.rnn import RNN


class TestRNN(unittest.TestCase):
    def setUp(self):
        # Define parameters for testing
        self.input_size = 10
        self.hidden_size = 20
        self.num_layers = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.activation = nn.Tanh()
        self.dropout = 0.2
        self.type = "GRU"
        # Create an instance of the RNN model
        self.model = RNN(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.device,
            self.activation,
            self.dropout,
            type=self.type,
        )

    def test_model_initialization(self):
        # Check if the model is an instance of nn.Module
        self.assertIsInstance(self.model, nn.Module)

        # Check if the model has the correct input size, hidden size, and number of layers
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.num_layers, self.num_layers)

        # Check if the device is set correctly
        self.assertEqual(self.model.device, self.device)

        # Check if the RNN layer is created with the correct parameters
        self.assertIsInstance(
            self.model.rnn,
            nn.RNN if self.type == "RNN" else nn.LSTM if self.type == "LSTM" else nn.GRU,
        )
        self.assertEqual(self.model.rnn.input_size, self.input_size)
        self.assertEqual(self.model.rnn.hidden_size, self.hidden_size)
        self.assertEqual(self.model.rnn.num_layers, self.num_layers)
        # self.assertEqual(self.model.rnn.nonlinearity, "tanh" if self.activation is isinstance(self.activation, nn.Tanh) else "relu")
        self.assertTrue(self.model.rnn.batch_first)
        self.assertEqual(self.model.rnn.dropout, self.dropout)

    def test_forward_pass(self):
        # Check if the forward pass runs without errors
        input_tensor = torch.randn(32, 5, self.input_size).to(self.device)
        output_tensor, hidden_tensor = self.model(input_tensor)
        self.assertEqual(output_tensor.shape, torch.Size([32, 5, self.hidden_size]))
        self.assertEqual(
            hidden_tensor.shape,
            torch.Size([self.num_layers, 32, self.hidden_size]),
        )


if __name__ == "__main__":
    unittest.main()
