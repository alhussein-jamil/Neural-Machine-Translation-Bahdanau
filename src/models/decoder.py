import torch.nn as nn
from models.rnn import RNN
from models.fcnn import FCNN
import torch
import torch.nn.functional as F
from typing import List, Dict, Any

class Alignment(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        device: str,
        activation: nn.Module = nn.Tanh(),
        last_layer_activation: nn.Module = nn.Identity(),
        dropout: float = 0
    ):
        super().__init__()
        self.nn = FCNN(
            input_size, hidden_sizes, output_size, device, activation, last_layer_activation, dropout
        )

    def forward(self, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Alignment module.

        Args:
            s (torch.Tensor): Tensor representing the context from the decoder.
            h (torch.Tensor): Tensor representing the hidden states from the encoder.

        Returns:
            torch.Tensor: Alignment vector.
        """
        # Find the alignment network response
        a = self.nn(torch.cat((s, h), dim=1))

        return a

class Decoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.alignment = Alignment(**config["alignment"])
        self.birnn = RNN(**config["birnn"])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder module.

        Args:
            h (torch.Tensor): Tensor representing the hidden states from the encoder.

        Returns:
            torch.Tensor: Tensor containing the predicted indices of the output tokens.
        """
        # h is the output of the encoder and has sizes (batch_size, Tx, hidden_size),
        # this function outputs a tensor of size (batch_size, Ty) containing the predicted indices of the output tokens
        # first, we need to find the context vector c
        # we do this by finding the alignment vector a
        output = torch.zeros(h.size(0), h.size(1), self.birnn.hidden_size).to(h.device)
        s = torch.zeros(
            self.birnn.num_layers * 1 if not self.birnn.rnn.bidirectional else 2,
            h.size(0),
            self.birnn.hidden_size,
        ).to(h.device)
        for i in range(h.size(1)):
            a = self.alignment(s.squeeze(0), h[:, i, :])
            e = F.softmax(a, dim=1)
            c = (h.swapaxes(1, 2) @ e.unsqueeze(2)).squeeze(2)  # c is of shape (batch_size, hidden_size)
            y, s = self.birnn(c.unsqueeze(1), s)
            output[:, i, :] = y.squeeze(1)
        return output
