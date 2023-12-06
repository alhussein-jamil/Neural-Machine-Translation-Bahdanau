import torch.nn as nn
from models.rnn import RNN
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder module for a Seq2Seq model.

    Args:
        input_size (int): The number of expected features in the input x.
        emb_size (int): The size of the embedding for input tokens.
        hidden_size (int): The number of features in the hidden state h.
        num_layers (int): Number of recurrent layers.
        device (str): Device to which the model is moved (e.g., 'cuda' or 'cpu').
        dropout_proba (float): Probability of dropout.

    Attributes:
        hidden_size (int): Number of features in the hidden state.
        embedding (nn.Embedding): Embedding layer for input tokens.
        rnn (RNN): Recurrent layer.
        dropout (nn.Dropout): Dropout layer.
        device (str): Device to which the model is moved.

    Methods:
        forward(source): Forward pass of the Encoder.
    """
    def __init__(self, input_size, emb_size, hidden_size, num_layers, device, dropout_proba):
        super().__init__()

        self.hidden_size = hidden_size      
        self.embedding = nn.Embedding(input_size, emb_size).to(device) 
        #input_size= emb_size
        self.rnn = RNN(emb_size, hidden_size, num_layers, device, dropout= dropout_proba)  
        self.dropout = nn.Dropout(dropout_proba).to(device)
        self.device = device
        
    def forward(self, source):   
        embedded = self.dropout(self.embedding(source.long().to(self.device)))
        #embedded = self.dropout(self.embedding(source))
        output, hidden = self.rnn(embedded)  
        return output, hidden

class Decoder(nn.Module):
    """
    Decoder module for a Seq2Seq model.

    Args:
        output_size (int): The number of expected features in the output.
        emb_size (int): The size of the embedding for output tokens.
        hidden_size (int): The number of features in the hidden state h.
        num_layers (int): Number of recurrent layers.
        device (str): Device to which the model is moved (e.g., 'cuda' or 'cpu').
        dropout_proba (float): Probability of dropout.

    Attributes:
        hidden_size (int): Number of features in the hidden state.
        output_size (int): The number of expected features in the output.
        embedding (nn.Embedding): Embedding layer for output tokens.
        rnn (RNN): Recurrent layer.
        fc_out (nn.Linear): Linear layer for output.
        dropout (nn.Dropout): Dropout layer.
        device (str): Device to which the model is moved.

    Methods:
        forward(input, hidden, context): Forward pass of the Decoder.
    """


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
        #self.context= context
        
    def forward(self, input, hidden,context):   
        # Ajoute une dimension pour traiter chaque élément de la séquence individuellement
        input = input.unsqueeze(0) # ajoute une dimension pour traiter chaque element de la séquence individuellement        
        
        # Conversion de l'indice en vecteur d'embedding avec dropout
        embedded = self.dropout(self.embedding(input))

        if context is None:
            context = torch.zeros_like(embedded) 

        # Concaténation de l'embedding et du contexte
        emb_con = torch.cat((embedded, context), dim = 2)     


        # Passage à travers la couche RNN avec mise à jour de l'état caché        
        output, hidden = self.rnn(emb_con)
        
        # Ajuster la dimensions
        hidden = hidden.squeeze(0)
        context = context.expand_as(hidden)
        embedded= embedded.expand_as(hidden) 

        print("Embedded shape:", embedded.squeeze(0).shape)
        print("Hidden shape:", hidden.squeeze(0).shape)
        print("Context shape:", context.squeeze(0).shape)

        # Concaténation de l'embedding, de l'état caché et du contexte pour la couche de sortie
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), 
                           dim = -1)
        
        # Prédiction avec la couche linéaire de sortie
        prediction = self.fc_out(output)  
        return prediction, hidden


class Encoder_RNNSearch(nn.Module):
    def __init__(self, input_size, emb_size, enc_hidden_size, dec_hidden_size,  num_layers, device, dropout_proba):
        super().__init__()       
        self.embedding = nn.Embedding(input_size, emb_size).to(device) 
        self.rnn = RNN(emb_size, enc_hidden_size,  num_layers, device,  dropout_proba, bidirectional = True)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size).to(device) 
        self.dropout = nn.Dropout(dropout_proba).to(device) 
        self.device = device
        
    def forward(self, source):
        embedded = self.dropout(self.embedding(source))  
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return outputs, hidden

class Decoder_RNNSearch(nn.Module):
    def __init__(self, output_size, emb_size, enc_hidden_size, dec_hidden_size, num_layers, device, dropout_proba, attention):
        super().__init__()

        # Définition des attributs
        self.output_size = output_size
        self.attention = attention   
        
        # Couche d'embedding pour convertir les indices en vecteurs
        self.embedding = nn.Embedding(output_size, emb_size).to(device) 

        # Couche RNN avec bidirectionnel et concaténation d'embedding et du contexte pondéré
        self.rnn = RNN((enc_hidden_size * 2) + emb_size, dec_hidden_size, num_layers, device,  dropout_proba, bidirectional = True)
        
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
    """
    Attention mechanism module.

    Args:
        enc_hidden_size (int): The number of features in the encoder's hidden state.
        dec_hidden_size (int): The number of features in the decoder's hidden state.
        device (str): Device to which the model is moved (e.g., 'cuda' or 'cpu').

    Attributes:
        attn (nn.Linear): Linear layer.
        v (nn.Linear): Linear layer.
        device (str): Device to which the model is moved.

    Methods:
        forward(hidden, encoder_outputs): Forward pass of the Attention.
    """
    
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
    

    





