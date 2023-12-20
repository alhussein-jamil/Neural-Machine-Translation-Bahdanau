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
        dropout = kwargs.get("dropout", 0.0)
        super().__init__()
        self.vocab_size = vocab_size
        # Utiliser la classe RNN dans Encoder
        self.rnn = RNN(
            input_size=embedding_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            device=rnn_device,
            activation=nn.Tanh(),
            dropout=dropout,
            bidirectional=True,
            type=rnn_type,
        )
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        # Appliquer l'embedding
        embedded = self.embedding(x.int()).float()
        # Appeler la classe RNN pour obtenir output et hidden
        rnn_output, rnn_hidden = self.rnn(embedded)
        return rnn_output.float(), rnn_hidden.float()
