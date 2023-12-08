from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fcnn import FCNN
from models.rnn import RNN


class Alignment(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.nn = FCNN(**kwargs)

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

class MaxoutUnit(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.neurons = FCNN(input_size,[],output_size,device)
    
    def forward(self, x):
        return torch.max(self.neurons(x), dim=1)[0] 

class Maxout(nn.Module):
    def __init__(self, input_size, output_size, num_units, device):
        super().__init__()
        self.num_units = num_units
        self.maxout_units = nn.ModuleList()
        for _ in range(num_units):
            self.maxout_units.append(MaxoutUnit(input_size, output_size, device))
    def forward(self, x):
        return torch.stack([maxout_unit(x) for maxout_unit in self.maxout_units], dim=1)

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.alignment = Alignment(**kwargs["alignment"])
        self.rnn = RNN(**kwargs["rnn"])
        self.maxout = Maxout(**kwargs["maxout"])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder module.

        Args:
            h (torch.Tensor): Tensor representing the hidden states from the encoder.

        Returns:
            torch.Tensor: Tensor containing the predicted indices of the output tokens.
        """
        # Initialize output tensor
        output = torch.zeros(h.size(0), h.size(1), self.maxout.num_units).to(h.device)

        # Initialize context vector
        s = torch.zeros(
            self.rnn.num_layers * (1 if not self.rnn.rnn.bidirectional else 2),
            h.size(0),
            self.rnn.hidden_size,
        ).to(h.device)

        for i in range(h.size(1)):
            # Compute alignment vector
            a = self.alignment(s.squeeze(0), h[:, i, :])
            e = F.softmax(a, dim=1)

            # Compute context vector
            c = torch.bmm(h.transpose(1, 2), e.unsqueeze(2)).squeeze(2)

            # Compute output and update context vector
            y, s = self.rnn(c.unsqueeze(1), s)
            maxed_out = self.maxout(y.squeeze(1))
            softmaxed = F.softmax(maxed_out, dim=1)

            output[:, i, :] = softmaxed


        return output