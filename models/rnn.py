import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device): 
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)