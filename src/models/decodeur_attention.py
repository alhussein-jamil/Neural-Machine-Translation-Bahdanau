import torch
import torch.nn as nn
import torch.nn.functional as F
import rnn

class Decoder(nn.Module):
    def __init__(self,  output_size, vocab_size=5, hidden_size=64, num_layers=1, device='cpu', dropout=0.0):
        super(Decoder, self).__init__()
        self.rnn = nn.RNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        ).to(device)

        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, annotations):

        # Calculate attention weights
        energies = self.v(torch.tanh(self.W_s(s) + self.W_c(annotations)))
        weights = F.softmax(energies, dim=1)

        # Calculate context vector
        context_vector = torch.bmm(weights.permute(0, 2, 1), annotations)

        # Concatenate the RNN output and the context vector
        concat_output = torch.cat([self.rnn, context_vector], dim=2)

        # Pass the concatenated vector through the linear layer
        output = self.out(concat_output)

        return output, s, weights

# Example usage:
hidden_size = 256
output_size = 10000  # Replace with your actual output vocabulary size
decoder = Decoder(hidden_size, output_size)

# Sample input for demonstration
input_token = torch.tensor([1])  # Replace with your actual input token
s_prev = torch.zeros(1, 1, hidden_size)  # Initial hidden state
annotations = torch.randn(1, 10, hidden_size)  # Replace with your actual annotations

# Forward pass through the decoder
output, s, weights = decoder(input_token, s_prev, annotations)

print("Output shape:", output.shape)
print("New hidden state shape:", s.shape)
print("Attention weights shape:", weights.shape)
