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
        self, input_size, hidden_size, num_layers, device, activation: nn.Module = nn.Tanh(), dropout=0, bidirectional=False
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

        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            nonlinearity="tanh" if activation is isinstance(activation, nn.Tanh) else "relu",
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        ).to(self.device)

    def forward(self, x):
        """
        Forward pass of the RNN.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # Forward propagate RNN
        out, hidden = self.rnn(x, h0)

        return out, hidden

class Encoder(nn.Module):
    def __init__(self, rnn_hidden_size,rnn_num_layers,rnn_device, vocab_size = 5):
        super().__init__()
        self.vocab_size = vocab_size
        #Utiliser la classe RNN dans Encoder
        self.rnn = RNN(
             input_size=vocab_size,
             hidden_size=rnn_hidden_size,
             num_layers=rnn_num_layers,
             device=rnn_device,
             activation=nn.Tanh(),
             dropout=0,
            bidirectional=True
         ) 
    
    # def vect_to_onehot: 

    def forward(self, x):
        k, t_x = x.shape


        v = torch.zeros(k, t_x, self.vocab_size)

        #Appliquer le one-hot coding
        v_one_hot = F.one_hot(x.long(), num_classes=self.vocab_size)

        # Initialize the hidden state
        num_directions = 2 if self.rnn.rnn.bidirectional else 1
        h0 = torch.zeros(self.rnn.num_layers * num_directions, k, self.rnn.hidden_size).to(self.rnn.device)


        # Appeler la classe RNN pour obtenir output et hidden
        rnn_output, rnn_hidden = self.rnn(v_one_hot)

        return rnn_output, rnn_hidden
        

class Decoder(nn.Module):
    def __init__(self, vocab_size=5, hidden_size=64, num_layers=1, device='cpu', dropout=0.0):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.rnn = nn.RNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        ).to(device)

        self.fc = nn.Linear(hidden_size, vocab_size).to(device)

    def forward(self, x, hidden):
        # x est la sortie du décodeur précédent (ou le mot cible au premier pas de temps)
        # hidden l'état caché de l'encodeur au premier pas de temps)

        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)

        return output, hidden


