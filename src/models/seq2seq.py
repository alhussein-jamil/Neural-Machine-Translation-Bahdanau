import torch.nn as nn
from rnn import RNN  # Importe un module RNN externe
import torch

# Définition de la classe Seq2Seq qui hérite de nn.Module
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder  # Initialise l'encodeur
        self.decoder = decoder  # Initialise le décodeur
        self.device = device    # Initialise le dispositif (CPU/GPU)
        
        # Vérifie que les dimensions cachées de l'encodeur et du décodeur sont égales
        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        
        # Vérifie que le nombre de couches de l'encodeur et du décodeur est égal
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    # Méthode de propagation avant (forward) du modèle
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
    
        batch_size = trg.shape[1]     # Taille du lot (batch)
        trg_len = trg.shape[0]        # Longueur de la séquence cible
        trg_vocab_size = self.decoder.output_dim  # Taille du vocabulaire de sortie du décodeur
        
        # Initialisation d'un tenseur pour stocker les sorties du décodeur
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # Obtenir l'état caché final et la cellule finale de l'encodeur
        hidden, cell = self.encoder(src)

        # Initialisation de la première entrée du décodeur avec le premier jeton de la séquence cible
        input = trg[0, :]
        
        # Boucle à travers la séquence cible
        for t in range(1, trg_len):
            # Propagation de l'entrée actuelle à travers le décodeur
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # Stockage de la sortie dans le tenseur de sorties
            outputs[t] = output
            
            # Décider d'utiliser le teacher forcing (alimentation forcée) ou non
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Obtenir le jeton prédit (top1) si pas d'alimentation forcée
            top1 = output.argmax(1) 
            
            # Utiliser soit le vrai jeton suivant ou le jeton prédit selon teacher_force
            input = trg[t] if teacher_force else top1
        
        return outputs
