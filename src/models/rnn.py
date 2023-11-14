import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, activation="tanh", dropout=0):

        super(RNN, self).__init__()
        
        self.device = device
        
        self.hidden_size = hidden_size
        
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity=activation, batch_first=True, dropout=dropout).to(self.device)  
        
    def forward(self, x):
            
            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
            
            # Forward propagate RNN
            out, _ = self.rnn(x, h0)
            
            return out