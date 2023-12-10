from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fcnn import FCNN
from models.rnn import RNN


class Alignment(nn.Module):
    def __init__(self, input_size: int, output_size: int, device: torch.device) -> None:
        super().__init__()
        self.hidden_size = input_size // 3

        self.nn_h = FCNN(input_size=self.hidden_size * 2, output_size=output_size, device=device)
        self.nn_s = FCNN(input_size=self.hidden_size, output_size=output_size, device=device)

    def forward(self, s_emb: torch.Tensor, h_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Alignment module.

        Args:
            s_emb (torch.Tensor): Tensor representing the context from the decoder.
            h_emb (torch.Tensor): Tensor representing the hidden states from the encoder.

        Returns:
            torch.Tensor: Alignment vector.
        """
        # Find the alignment network response
        a = F.tanh(s_emb + h_emb)

        return a


class OutputNetwork(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        max_out_units: int,
        hidden_size: int,
        vocab_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.t_nn = FCNN(
            input_size=embedding_size + 3 * hidden_size,
            hidden_sizes=[],
            output_size=2 * max_out_units,
            device=device,
        )
        self.output_nn = FCNN(
            input_size=max_out_units,
            hidden_sizes=[],
            output_size=vocab_size,
            device=device,
        )
        self.output_size = vocab_size

    def forward(self, s_i: torch.Tensor, y_i: torch.Tensor, c_i: torch.Tensor) -> torch.Tensor:
        # based on the article Maxout Networks
        t_tilde = self.t_nn(torch.cat((s_i, y_i, c_i), dim=1))
        t_even = t_tilde[:, : t_tilde.size(1) // 2]
        t_odd = t_tilde[:, t_tilde.size(1) // 2 :]
        t = torch.max(t_even, t_odd)
        return self.output_nn(t)


class Decoder(nn.Module):
    def __init__(self, alignment: Dict[str, Any], rnn: Dict[str, Any], embedding: Dict[str, Any], output_nn: Dict[str, Any]) -> None:
        super().__init__()

        self.alignment = Alignment(**alignment)
        self.rnn = RNN(**rnn)
        self.embedding = FCNN(
            input_size=rnn["hidden_size"],
            output_size=embedding["embedding_size"],
            device=embedding["device"],
        )
        self.hidden_size = rnn["hidden_size"]
        self.output_nn = OutputNetwork(**output_nn)

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

        h_emb = self.alignment.nn_h(h)

        for i in range(h.size(1)):
            s_i_emb = self.alignment.nn_s(s_i.view(h.size(0), -1))

            # Compute alignment vector
            a = self.alignment(s_i_emb, h_emb[:, i, :])

            e = F.softmax(a, dim=1)

            # Compute context vector
            c = torch.bmm(h.transpose(1, 2), e.unsqueeze(2)).squeeze(2)

            # Compute output and update context vector
            raw_y_i, s_i = self.rnn(c.unsqueeze(1), s_i)

            embed_y_i = self.embedding(raw_y_i.squeeze(1))

            output_network_out = self.output_nn(s_i.view(h.size(0), -1), embed_y_i.squeeze(1), c)

            output[:, i, :] = output_network_out

        return output
