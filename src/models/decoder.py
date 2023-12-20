from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fcnn import FCNN
from models.rnn import RNN


class Alignment(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        device: torch.device,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = input_size // 3

        self.nn_h = FCNN(
            input_size=self.hidden_size * 2,
            output_size=self.hidden_size,
            device=device,
            dropout=dropout,
            mean=0,
            std=0.001,
        )
        self.nn_s = FCNN(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            device=device,
            dropout=dropout,
            mean=0,
            std=0.001,
        )
        self.nn_v = FCNN(
            input_size=self.hidden_size,
            output_size=output_size,
            device=device,
            dropout=dropout,
            mean=0,
            std=0.0,
        )

    def forward(self, s_emb: torch.Tensor, h_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Alignment module.

        Args:
            s_emb (torch.Tensor): Tensor representing the context from the decoder.
            h_emb (torch.Tensor): Tensor representing the hidden states from the encoder.

        Returns:
            torch.Tensor: Alignment vector.
        """
        return self.nn_v(F.tanh(s_emb + h_emb)).float()

    def forward_unoptimized(self, s, h):
        return self.forward(self, self.nn_s(s), self.nn_h(h)).float()


class OutputNetwork(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        max_out_units: int,
        hidden_size: int,
        vocab_size: int,
        device: torch.device,
        **kwargs,
    ) -> None:
        """
        Initializes the OutputNetwork module.

        Args:
            embedding_size (int): The size of the input embeddings.
            max_out_units (int): The maximum number of output units.
            hidden_size (int): The size of the hidden layers.
            vocab_size (int): The size of the vocabulary.
            device (torch.device): The device to be used for computation.
        """
        super().__init__()

        # matrix for s_i
        self.u_o = FCNN(
            input_size=hidden_size,
            output_size=2 * max_out_units,
            **kwargs,
        )

        # matrix for y_i
        self.v_o = FCNN(
            input_size=embedding_size,
            output_size=2 * max_out_units,
            **kwargs,
        )

        # matrix for c_i
        self.c_o = FCNN(
            input_size=2 * hidden_size,
            output_size=2 * max_out_units,
            **kwargs,
        )

        self.output_nn = FCNN(
            input_size=max_out_units,
            output_size=vocab_size,
            device=device,
            **kwargs,
        )
        self.output_size = vocab_size

    def forward(
        self, s_i: torch.Tensor, y_i: torch.Tensor, c_i: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a forward pass through the OutputNetwork module.

        Args:
            s_i (torch.Tensor): The input tensor s_i.
            y_i (torch.Tensor): The input tensor y_i.
            c_i (torch.Tensor): The input tensor c_i.

        Returns:
            torch.Tensor: The output tensor.
        """
        # based on the article Maxout Networks
        t_tilde =(self.u_o(s_i) + self.v_o(y_i) + self.c_o(c_i)).float()

        # sep odd and even
        t_even = t_tilde[:, 0::2]
        t_odd = t_tilde[:, 1::2]

        return self.output_nn(torch.max(t_even, t_odd))


class Decoder(nn.Module):
    def __init__(
        self,
        alignment: Dict[str, Any],
        rnn: Dict[str, Any],
        embedding: Dict[str, Any],
        output_nn: Dict[str, Any],
        traditional: bool = False,
    ) -> None:
        super().__init__()

        # specify if we use the traditional encoder-decoder model
        self.traditional = traditional

        if traditional:
            rnn["input_size"] = rnn["hidden_size"] * 2

        # Alignment module
        self.alignment = Alignment(**alignment)

        self.hidden_size = rnn["hidden_size"]

        # GRU layer
        self.rnn = RNN(**rnn)

        # Embedding layer
        self.embedding = FCNN(
            input_size=rnn["hidden_size"],
            output_size=embedding["embedding_size"],
            device=embedding["device"],
        )
        # self.batch_norm_enc = nn.BatchNorm1d(2*rnn["hidden_size"])
        self.output_nn = OutputNetwork(**output_nn)

        # Used for the traditional model to return to the output size
        self.relaxation_nn = FCNN(
            input_size=rnn["hidden_size"],
            output_size=output_nn["vocab_size"],
        )

        # Initialize context vector as a learnable parameter
        self.Ws = nn.Linear(rnn["hidden_size"], rnn["hidden_size"]).to(
            embedding["device"]
        )
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder module.

        Args:
            h (torch.Tensor): Tensor representing the hidden states from the encoder.

        Returns:
            torch.Tensor: Tensor containing the predicted indices of the output tokens.
        """
        # h = self.batch_norm_enc(h.reshape(-1, 2*self.hidden_size)).reshape(h.shape[0], -1, 2*self.hidden_size)

        # Initialize output tensor
        output = torch.zeros(h.size(0), h.size(1), self.output_nn.output_size, device=h.device,dtype=torch.float16)

        if not self.traditional:
            # Initialize context vector as a learnable parameter
            s_i = F.tanh(self.Ws(h[:, 0, self.rnn.hidden_size :])).view(
                self.rnn.num_layers * (1 if not self.rnn.rnn.bidirectional else 2),
                h.size(0),
                self.rnn.hidden_size,
            ).float()

            embed_y_i = torch.zeros(h.size(0), self.embedding.output_size, device=h.device,dtype=torch.float16)

            h_emb = self.alignment.nn_h(h)
            allignments = []
            for i in range(h.size(1)):
                # Compute the embedding of the current context vector
                s_i_emb = self.alignment.nn_s(s_i.view(h.size(0), -1))

                # Compute alignment vector
                a = self.alignment(s_i_emb, h_emb[:, i, :])
                allignments.append(a)
                # Apply softmax to obtain attention weights
                e = F.softmax(a, dim=1)
                # Compute context vector
                c = torch.bmm(h.transpose(1, 2), e.unsqueeze(2)).squeeze(2).float()

                # Compute output and update context vector
                raw_y_i, s_i = self.rnn(
                    torch.cat((embed_y_i.unsqueeze(1), c.unsqueeze(1)), dim=2), s_i
                )

                # Embed the output token
                embed_y_i = self.embedding(raw_y_i.squeeze(1))

                # Compute the output of the output network
                output_network_out = self.output_nn(
                    s_i.view(h.size(0), -1), embed_y_i.squeeze(1), c
                )

                # Store the output in the output tensor
                output[:, i, :] = output_network_out
        else:
            output_rnn, _ = self.rnn(h)
            relaxed = self.relaxation_nn(output_rnn)
            output[:, :, :] = relaxed

        return output, torch.stack(
            allignments, dim=1
        ) if not self.traditional else torch.zeros(h.shape[0], h.shape[1], h.shape[1])
