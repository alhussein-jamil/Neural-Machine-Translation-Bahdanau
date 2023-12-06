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
        
'''      
en = Encoder(vocab_size=5)
print(en(torch.tensor([
    (4,2,1),
    (2,1,0)
])))'''

<<<<<<< Updated upstream
en = Encoder(
    rnn_hidden_size=16,
    rnn_num_layers=1,
    rnn_device='cpu',  # Vous pouvez spécifier le périphérique approprié ici
    vocab_size=5
)

# Définir une séquence d'entrée (utiliser des listes au lieu de tuples)
sequence_input = torch.tensor([
    [4, 2, 1],
    [2, 1, 0]
])

# Appeler l'encodeur pour obtenir le vecteur c
encoded_output, encoded_hidden = en(sequence_input)

# Afficher la sortie encodée et l'état caché
print("Encoded Output:")
print(encoded_output)
print("Encoded Hidden State:")
print(encoded_hidden)
=======
en(torch.rand(3,5))

>>>>>>> Stashed changes


class Encoder(nn.Module):
    def __init__(self,device, vocab_size = 5 ,hidden_size= 64, num_layers= 64 , dropout=0, bidirectional= True):

        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size=hidden_size,
        self.num_layers=num_layers,
        self.device=device,
        self.dropout= dropout.to(device)
        self.rnn = RNN( hidden_size, num_layers, device, dropout) 

    def forward(self, x):
        k, t_x = x.shape


        v = torch.zeros(k, t_x, self.vocab_size)

        #Appliquer le one-hot coding
        v_one_hot = F.one_hot(x.long(), num_classes=self.vocab_size)

        # Appeler la classe RNN pour obtenir output et hidden
         

        rnn_output, rnn_hidden = self.rnn(v_one_hot)

        return rnn_output, rnn_hidden
        
en = Encoder(vocab_size=5)
print(en(torch.tensor([
    (4,2,1),
    (2,1,0)
])))
