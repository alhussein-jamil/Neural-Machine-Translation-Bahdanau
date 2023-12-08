from typing import Any, Dict, List, Tuple

import torch
from torch import nn


class FCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        device: str,
        activation: nn.Module = nn.Tanh(),
        last_layer_activation: nn.Module = nn.Identity(),
        dropout: float = 0,
    ):
        super(FCNN, self).__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.last_layer_activation = last_layer_activation

        # Create a list of fully-connected layers
        self.fc = nn.ModuleList()

        # Add the first fully-connected layer
        self.fc.append(nn.Linear(self.input_size, self.hidden_sizes[0] if len(self.hidden_sizes) > 0 else self.output_size))

        # Add the remaining fully-connected layers
        for i in range(1, len(self.hidden_sizes)):
            self.fc.append(nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))


        if len(self.hidden_sizes) > 0:
            # Add the final fully-connected layer
            self.fc.append(nn.Linear(self.hidden_sizes[-1], self.output_size))

        self.fc = self.fc.to(self.device)

        # Add the activation function
        self.activation = activation

        # Add the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FCNN.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Iterate through the fully-connected layers
        for i in range(len(self.fc) - 1):
            # Apply the linear transformation
            x = self.fc[i](x)

            # Apply the activation function
            x = self.activation(x)

            # Apply the dropout layer
            x = self.dropout(x)

        # Apply the final fully-connected layer
        x = self.fc[-1](x)
        x = self.last_layer_activation(x)

        return x
