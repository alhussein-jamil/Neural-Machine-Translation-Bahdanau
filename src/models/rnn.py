from torch import nn
import torch
import torch.nn.functional as F 

class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN) Module.

    Args:
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        num_layers (int): Number of recurrent layers.
        device (str): Device to which the model is moved (e.g., 'cuda' or 'cpu').
        activation (str, optional): Type of nonlinearity. Default is 'tanh'.
        dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer. Default is 0.

    Attributes:
        device (str): Device to which the model is moved.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        rnn (torch.nn.RNN): RNN layer.

    Methods:
        forward(x): Forward pass of the RNN.

    """

    def __init__(
        self, input_size, hidden_size, num_layers, device, activation: nn.Module = nn.Tanh(), dropout=0, bidirectional=False, type = 'RNN'
    ):
        """
        Initializes the RNN module.

        Parameters:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers.
            device (str): Device to which the model is moved (e.g., 'cuda' or 'cpu').
            activation (str, optional): Type of nonlinearity. Default is 'tanh'.
            dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer. Default is 0.
        """
        super(RNN, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        network_type = nn.RNN if type == 'RNN' else nn.LSTM if type == 'LSTM' else nn.GRU
        self.rnn = network_type(
            input_size,
            hidden_size,
            num_layers,
            # nonlinearity="tanh" if activation is isinstance(activation, nn.Tanh) else "relu",
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        ).to(self.device)

    def forward(self, x, h0 = None):
        """
        Forward pass of the RNN.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Initialize hidden state
        if h0 is None:
            h0 = torch.zeros( self.num_layers * 1 if not self.rnn.bidirectional else 2, x.size(0), self.hidden_size).to(
                self.device
            )

        #Forward propagate RNN
        out, hidden = self.rnn(x,h0)
        return out, hidden



        
# '''      
# en = Encoder(vocab_size=5)
# print(en(torch.tensor([
#     (4,2,1),
#     (2,1,0)
# ])))'''

# en = Encoder(
#     rnn_hidden_size=16,
#     rnn_num_layers=1,
#     rnn_device='cpu',  # Vous pouvez spécifier le périphérique approprié ici
#     vocab_size=5
# )

# # Définir une séquence d'entrée (utiliser des listes au lieu de tuples)
# sequence_input = torch.tensor([
#     [4, 2, 1],
#     [2, 1, 0]
# ])

# # Appeler l'encodeur pour obtenir le vecteur c
# encoded_output, encoded_hidden = en(sequence_input)

# # Afficher la sortie encodée et l'état caché
# print("Encoded Output:")
# print(encoded_output)
# print("Encoded Hidden State:")
# print(encoded_hidden)
        

# class Decoder(nn.Module):
#     def __init__(self, vocab_size=5, hidden_size=64, num_layers=1, device='cpu', dropout=0.0):
#         super(Decoder, self).__init__()

#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.device = device

#         self.rnn = nn.RNN(
#             input_size=vocab_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout,
#         ).to(device)

#         self.fc = nn.Linear(hidden_size, vocab_size).to(device)

#     def forward(self, x, hidden):
#         # x est la sortie du décodeur précédent (ou le mot cible au premier pas de temps)
#         # hidden l'état caché de l'encodeur au premier pas de temps)

#         output, hidden = self.rnn(x, hidden)
#         output = self.fc(output)

#         return output, hidden



