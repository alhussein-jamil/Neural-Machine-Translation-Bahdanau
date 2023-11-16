from torch import nn
import torch


class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN) Module.

    Args:
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        num_layers (int): Number of recurrent layers.
        device (str): Device to which the model is moved (e.g., 'cuda' or 'cpu').
        activation (str, optional): Type of nonlinearity. Default is 'tanh'.
        dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer. Default is 0.

    Attributes:
        device (str): Device to which the model is moved.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        rnn (torch.nn.RNN): RNN layer.

    Methods:
        forward(x): Forward pass of the RNN.

    """

    def __init__(
        self, input_size, hidden_size, num_layers, device, activation="tanh", dropout=0
    ):
        """
        Initializes the RNN module.

        Parameters:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers.
            device (str): Device to which the model is moved (e.g., 'cuda' or 'cpu').
            activation (str, optional): Type of nonlinearity. Default is 'tanh'.
            dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer. Default is 0.
        """
        super(RNN, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            nonlinearity=activation,
            batch_first=True,
            dropout=dropout,
        ).to(self.device)

    def forward(self, x):
        """
        Forward pass of the RNN.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        return out
