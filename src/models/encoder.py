import torch.nn as nn
import torch.nn.functional as F

from models.fcnn import FCNN
from models.rnn import RNN


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        rnn_hidden_size = kwargs.get("rnn_hidden_size", 5)
        rnn_num_layers = kwargs.get("rnn_num_layers", 1)
        rnn_device = kwargs.get("rnn_device", "cpu")
        vocab_size = kwargs.get("vocab_size", 5)
        rnn_type = kwargs.get("rnn_type", "GRU")
        embedding_size = kwargs.get("embedding_size", 5)

        super().__init__()
        self.vocab_size = vocab_size
        # Utiliser la classe RNN dans Encoder
        self.rnn = RNN(
            input_size=embedding_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            device=rnn_device,
            activation=nn.Tanh(),
            dropout=0,
            bidirectional=True,
            type=rnn_type,
        )
        self.embedding = FCNN(
            input_size=vocab_size,
            output_size=embedding_size,
            device=rnn_device,
            activation=nn.Tanh(),
            last_layer_activation=nn.Identity(),
            dropout=0,
        )

    def forward(self, x):
        # Appliquer le one-hot coding
        v_one_hot = F.one_hot(x.long(), num_classes=self.vocab_size).float()

        # Appliquer l'embedding
        embedded = self.embedding(v_one_hot.view(-1, self.vocab_size)).view(v_one_hot.shape[0], v_one_hot.shape[1], -1)
        # Appeler la classe RNN pour obtenir output et hidden
        rnn_output, rnn_hidden = self.rnn(embedded)

        return rnn_output, rnn_hidden
