import unittest

import torch
from torch import nn

from models.fcnn import FCNN


class TestFCNN(unittest.TestCase):
    def setUp(self):
        # Define parameters for testing
        self.input_size = 10
        self.hidden_sizes = [20, 30]
        self.output_size = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.activation = nn.ReLU()
        self.last_layer_activation = nn.Sigmoid()
        self.dropout = 0.2

        # Create an instance of the FCNN model
        self.model = FCNN(
            self.input_size,
            self.output_size,
            self.hidden_sizes,
            self.device,
            self.activation,
            self.last_layer_activation,
            self.dropout,
        ).to(self.device)

    def test_model_initialization(self):
        # Check if the model is an instance of nn.Module
        self.assertIsInstance(self.model, nn.Module)

        # Check if the model has the correct input size, hidden sizes, and output size
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_sizes, self.hidden_sizes)
        self.assertEqual(self.model.output_size, self.output_size)

        # Check if the device is set correctly
        self.assertEqual(str(self.model.device), self.device)

        # Check if the activation functions and dropout are set correctly
        self.assertEqual(self.model.activation, self.activation)
        self.assertEqual(self.model.last_layer_activation, self.last_layer_activation)
        self.assertEqual(self.model.dropout.p, self.dropout)

    def test_forward_pass(self):
        # Check if the forward pass runs without errors
        input_tensor = torch.randn(32, self.input_size).to(self.device)
        output_tensor = self.model(input_tensor)
        self.assertEqual(output_tensor.shape, torch.Size([32, self.output_size]))

    def test_parameters_on_device(self):
        # Check if the model parameters are on the correct device
        for param in self.model.parameters():
            assert self.device in str(param.device)


if __name__ == "__main__":
    unittest.main()
