import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedHiddenUnit(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GatedHiddenUnit, self).__init__()
        self.Wz = nn.Linear(input_size, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size)
        self.Cz = nn.Linear(hidden_size, hidden_size)

        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.Cr = nn.Linear(hidden_size, hidden_size)

        self.We = nn.Linear(input_size, hidden_size)

    def forward(self, xi, si_1, ci):
        zi = torch.sigmoid(self.Wz(xi) + self.Uz(si_1) + self.Cz(ci))
        ri = torch.sigmoid(self.Wr(xi) + self.Ur(si_1) + self.Cr(ci))
        s_hat_i = torch.tanh(self.We(xi) + self.Ur(ri * si_1) + ci)
        si = (1 - zi) * si_1 + zi * s_hat_i
        return si

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gated_hidden_units = nn.ModuleList(
            [GatedHiddenUnit(input_size, hidden_size) for _ in range(num_layers)]
        )

    def forward(self, x, h0, c0):
        # x: input sequence, h0: initial hidden state, c0: initial cell state
        si = h0  # initialize si (hidden state)
        ci = c0  # initialize ci (cell state)
        for layer in self.gated_hidden_units:
            si = layer(x, si, ci)
            ci = si  # For simplicity, use si as the new cell state

        return si, ci

class GatedRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        device,
        activation: nn.Module = nn.Tanh(),
        dropout=0,
        bidirectional=False,
    ):
        super(GatedRNN, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.embedding = nn.Embedding(input_size, hidden_size)  # Ajout de la couche d'embedding
        self.custom_lstm = CustomLSTM(hidden_size, hidden_size, num_layers)  # Notez le changement ici

    def forward(self, x, h0=None, c0=None):
        # x: input sequence, h0: initial hidden state, c0: initial cell state
        x_embedded = self.embedding(x)  # Appliquer l'embedding
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers * 1 if not self.custom_lstm.bidirectional else 2,
                x.size(0),
                self.hidden_size,
            ).to(self.device)

        if c0 is None:
            c0 = torch.zeros(
                self.num_layers * 1 if not self.custom_lstm.bidirectional else 2,
                x.size(0),
                self.hidden_size,
            ).to(self.device)

        out, _ = self.custom_lstm(x_embedded, h0, c0)
        return out

# Exemple
vocab_size = 10000 
input_size = vocab_size
hidden_size = 20
num_layers = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

model = GatedRNN(input_size, hidden_size, num_layers, device)
