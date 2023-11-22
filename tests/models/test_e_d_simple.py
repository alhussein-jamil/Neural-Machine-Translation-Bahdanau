#test
from models.enco_deco import Encoder, Decoder
import torch

input_size = 1000
emb_size = 256
hidden_size = 512
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout_proba = 0.5

# Créer une instance de l'encodeur
encoder = Encoder(input_size, emb_size, hidden_size, num_layers, device, dropout_proba)

# Générer des données d'entrée simulées
sequence_length = 10
batch_size = 32
input_data = torch.randint(0, input_size, (sequence_length, batch_size)).to(device)

# Passer les données d'entrée à l'encodeur
hidden_state = encoder(input_data)

# Afficher la taille du tenseur résultant
print("Hidden state size:", hidden_state.size())



