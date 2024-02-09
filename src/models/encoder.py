import torch.nn as nn

from models.rnn import RNN
from global_variables import DEVICE
import torch
from models.fcnn import FCNN

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
            dropout=dropout,
            bidirectional=True,
            type=rnn_type,
        )
        self.embedding = FCNN(
            input_size=vocab_size,
            output_size=embedding_size,
            device=rnn_device,
            dropout=dropout,
        )

    @torch.autocast(DEVICE)
    def forward(self, x):
        x = torch.nn.functional.one_hot(x.long(), self.vocab_size).half()   
        # Appliquer l'embedding
        embedded = self.embedding(x.float())
        # Appeler la classe RNN pour obtenir output et hidden
        with torch.autocast(DEVICE):
            rnn_output, rnn_hidden = self.rnn(embedded)
        return rnn_output, rnn_hidden
