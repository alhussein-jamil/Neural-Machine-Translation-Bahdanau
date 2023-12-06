import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rnn import RNN


class Encoder(nn.Module):
    def __init__(
        self, rnn_hidden_size, rnn_num_layers, rnn_device, vocab_size=5, rnn_type="GRU"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        # Utiliser la classe RNN dans Encoder
        self.rnn = RNN(
            input_size=vocab_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            device=rnn_device,
            activation=nn.Tanh(),
            dropout=0,
            bidirectional=True,
            type=rnn_type,
        )

    def forward(self, x):
        # Appliquer le one-hot coding
        v_one_hot = F.one_hot(x.long(), num_classes=self.vocab_size)

        # Appeler la classe RNN pour obtenir output et hidden
        rnn_output, rnn_hidden = self.rnn(v_one_hot)

        return rnn_output, rnn_hidden
