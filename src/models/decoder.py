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
        self.neurons = FCNN(input_size, [], output_size, device)

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
        return torch.stack([maxout_unit(x) for maxout_unit in self.maxout_unit], dim=1)


class OutputNetwork(nn.Module):
    def __init__(self, embedding_size, max_out_units, hidden_size,vocab_size, device):
        super().__init__()
        self.t_nn = FCNN(input_size=embedding_size + 3 * hidden_size, hidden_sizes=[], output_size=2 * max_out_units, device=device)
        self.output_nn = FCNN(input_size=max_out_units, hidden_sizes=[], output_size=vocab_size, device=device)
        self.output_size = vocab_size

    def forward(self, s_i, y_i, c_i):
        t_tilde = self.t_nn(torch.cat((s_i, y_i, c_i), dim=1))
        t_even = t_tilde[:, :t_tilde.size(1) // 2]
        t_odd = t_tilde[:, t_tilde.size(1) // 2:]
        t = torch.max(t_even, t_odd)
        return self.output_nn(t)


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.alignment = Alignment(**kwargs["alignment"])
        self.rnn = RNN(**kwargs["rnn"])
        # self.maxout = Maxout(**kwargs["maxout"])
        # self.fcnn = FCNN(**kwargs["fcnn"])
        self.embedding = FCNN(
            input_size = kwargs["rnn"]["hidden_size"],
            output_size= kwargs["embedding"]["embedding_size"],
        )
        self.output_nn = OutputNetwork(**kwargs["output_nn"])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder module.

        Args:
            h (torch.Tensor): Tensor representing the hidden states from the encoder.

        Returns:
            torch.Tensor: Tensor containing the predicted indices of the output tokens.
        """
        # Initialize output tensor
        output = torch.zeros(h.size(0), h.size(1), self.output_nn.output_size).to(h.device)

        # Initialize context vector
        s_i = torch.zeros(
            self.rnn.num_layers * (1 if not self.rnn.rnn.bidirectional else 2),
            h.size(0),
            self.rnn.hidden_size,
        ).to(h.device)

        for i in range(h.size(1)):
            # Compute alignment vector
            a = self.alignment(s_i.squeeze(0), h[:, i, :])
            e = F.softmax(a, dim=1)

            # Compute context vector
            c = torch.bmm(h.transpose(1, 2), e.unsqueeze(2)).squeeze(2)

            # Compute output and update context vector
            raw_y_i, s_i = self.rnn(c.unsqueeze(1), s_i)
            embed_y_i = self.embedding(raw_y_i.squeeze(1))
            output_network_out = self.output_nn(s_i.view(h.size(0), -1), embed_y_i.squeeze(1), c)
            # maxed_out = self.maxout(y.squeeze(1))
            # fcnn_out = self.fcnn(embed_y_i.squeeze(1))
            # softmaxed = F.softmax(maxed_out, dim=1)
            # softmaxed = F.log_softmax(fcnn_out, dim=1)
            softmaxed = F.log_softmax(output_network_out, dim=1)
            output[:, i, :] = softmaxed

        return output


