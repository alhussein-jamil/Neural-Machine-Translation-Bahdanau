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

######### DECODEUR
output_size = 20
emb_size = 10
hidden_size = 10
num_layers = 2
dropout_proba = 0.1

# Créer une instance du décodeur
decoder = Decoder(output_size, emb_size, hidden_size, num_layers, device, dropout_proba)

# Définir des données factices pour le test
input_data = torch.tensor([1])  
hidden_state = torch.zeros(num_layers, 1, hidden_size).to(device) 
context_data = torch.zeros(1, 1, emb_size).to(device)  

# Passer les données à travers le décodeur
output, new_hidden_state = decoder(input_data, hidden_state, context_data)

# Afficher les résultats
print("Output shape:", output.shape)
print("New hidden state shape:", new_hidden_state.shape)



# Paramètres du modèle
output_size = 100  # Remplacez par la taille de votre vocabulaire de sortie
emb_size = 50
hidden_size = 64
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout_proba = 0.1

# Création du décodeur
decoder = Decoder(output_size, emb_size, hidden_size, num_layers, device, dropout_proba)

# Test avec une séquence factice
input_sequence = torch.tensor([1, 2, 3, 4]).unsqueeze(0).to(device)  
initial_hidden = torch.zeros(num_layers, 1, hidden_size).to(device)

# Forward pass
output, new_hidden = decoder(input_sequence, initial_hidden, None)

# Affichage des résultats
print("Input sequence:", input_sequence)
print("Output sequence:", torch.argmax(output, dim=2))
print("New hidden state:", new_hidden)
