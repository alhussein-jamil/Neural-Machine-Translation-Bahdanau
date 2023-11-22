import torch.nn as nn
from models.rnn import RNN
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layers, device, dropout_proba):
        super().__init__()

        self.hidden_size = hidden_size      
        self.embedding = nn.Embedding(input_size, emb_size).to(device) 
        self.rnn = RNN(emb_size, hidden_size, num_layers, device, dropout= dropout_proba)  
        self.dropout = nn.Dropout(dropout_proba).to(device)
        self.device = device
        
    def forward(self, source):               
        embedded = self.dropout(self.embedding(source))
        output, hidden = self.rnn(embedded)     
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size, num_layers, device, dropout_proba):
        super().__init__()
        # Définition des attributs
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Couche d'embedding pour convertir les indices en vecteurs
        self.embedding = nn.Embedding(output_size, emb_size).to(device) 

        # Couche RNN avec concaténation d'embedding et de l'état caché précédent
        self.rnn = RNN(emb_size + hidden_size, hidden_size, num_layers, device ,dropout=dropout_proba)

        # Couche linéaire de sortie avec concaténation d'embedding, de l'état caché et du contexte
        self.fc_out = nn.Linear(emb_size + hidden_size * 2, output_size).to(device) 

        # Couche de dropout pour la régularisation
        self.dropout = nn.Dropout(dropout_proba).to(device) 
        self.device = device
        
    def forward(self, input, hidden, context):   
        # Ajoute une dimension pour traiter chaque élément de la séquence individuellement
        input = input.unsqueeze(0) # ajoute une dimension pour traiter chaque element de la séquence individuellement        
        
        # Conversion de l'indice en vecteur d'embedding avec dropout
        embedded = self.dropout(self.embedding(input))
        if context is None:
            context = torch.zeros_like(embedded)  # Ou tout autre mécanisme d'initialisation


        # Concaténation de l'embedding et du contexte
        emb_con = torch.cat((embedded, context), dim = 2)    

        # Passage à travers la couche RNN avec mise à jour de l'état caché        
        output, hidden = self.rnn(emb_con, hidden)

        # Concaténation de l'embedding, de l'état caché et du contexte pour la couche de sortie
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), 
                           dim = 1)
        
        # Prédiction avec la couche linéaire de sortie
        prediction = self.fc_out(output)  
        return prediction, hidden


class Encoder_RNNSearch(nn.Module):
    def __init__(self, input_size, emb_size, enc_hidden_size, dec_hidden_size,  num_layers, device, dropout_proba, activation="tanh"):
        super().__init__()       
        self.embedding = nn.Embedding(input_size, emb_size).to(device) 
        self.rnn = RNN(emb_size, enc_hidden_size,  num_layers, device,  dropout_proba, bidirectional = True, activation="tanh")
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size).to(device) 
        self.dropout = nn.Dropout(dropout_proba).to(device) 
        self.device = device
        
    def forward(self, source):
        embedded = self.dropout(self.embedding(source))  
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return outputs, hidden

class Decoder_RNNSearch(nn.Module):
    def __init__(self, output_size, emb_size, enc_hidden_size, dec_hidden_size, num_layers, device, dropout_proba, attention, activation="tanh"):
        super().__init__()

        # Définition des attributs
        self.output_size = output_size
        self.attention = attention   
        
        # Couche d'embedding pour convertir les indices en vecteurs
        self.embedding = nn.Embedding(output_size, emb_size).to(device) 

        # Couche RNN avec bidirectionnel et concaténation d'embedding et du contexte pondéré
        self.rnn = RNN((enc_hidden_size * 2) + emb_size, dec_hidden_size, num_layers, device,  dropout_proba, bidirectional = True, activation="tanh")
        
        # Couche linéaire de sortie avec concaténation d'embedding, de l'état caché et du contexte pondéré
        self.fc_out = nn.Linear((enc_hidden_size * 2) + dec_hidden_size + emb_size, output_size).to(device) 

        # Couche de dropout pour la régularisation
        self.dropout = nn.Dropout(dropout_proba).to(device) 
        self.device = device
        
    def forward(self, input, hidden, encoder_outputs):          
        # Ajoute une dimension pour traiter chaque élément de la séquence individuellement
        input = input.unsqueeze(0) 

        # Conversion de l'indice en vecteur d'embedding avec dropout
        embedded = self.dropout(self.embedding(input))    

        # Calcul de l'attention sur les sorties de l'encodeur 
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)

        # Concaténation de l'embedding et du contexte pondéré
        rnn_input = torch.cat((embedded, weighted), dim = 2)

        # Passage à travers la couche RNN avec mise à jour de l'état caché
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        assert (output == hidden).all() 

        # Réduction des dimensions pour la couche de sortie
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0) 

        # Prédiction avec la couche linéaire de sortie
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        return prediction, hidden.squeeze(0)

class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, device):
        super().__init__()      

        # Couches linéaires pour le calcul de l'attention
        self.attn = nn.Linear((enc_hidden_size * 2) + dec_hidden_size, dec_hidden_size).to(device)
        self.v = nn.Linear(dec_hidden_size, 1, bias = False).to(device)
        self.device = device

    def forward(self, hidden, encoder_outputs):
                # Répétition du vecteur caché pour correspondre aux dimensions des sorties de l'encodeur

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # Calcul de l'énergie et de l'attention
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 

        # Normalisation de l'attention avec softmax
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)
    

    





