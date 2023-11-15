import torch.nn as nn
from rnn import RNN
import torch

class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layers, device, dropout_proba):
        super().__init__()

        self.hidden_size = hidden_size      
        self.embedding = nn.Embedding(input_size, emb_size)     
        self.rnn = RNN(emb_size, hidden_size, num_layers, device)  
        self.dropout = nn.Dropout(dropout_proba)
        
    def forward(self, source):               
        embedded = self.dropout(self.embedding(source))
        outputs, hidden = self.rnn(embedded)     
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size, num_layers, device, dropout_proba):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size      
        self.embedding = nn.Embedding(output_size, emb_size)
        self.rnn = RNN(emb_size + hidden_size, hidden_size, num_layers, device)
        self.fc_out = nn.Linear(emb_size + hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout_proba)
        
    def forward(self, input, hidden, context):   
        input = input.unsqueeze(0) # ajoute une dimension pour traiter chaque element de la séquence individuellement        
        embedded = self.dropout(self.embedding(input))
        emb_con = torch.cat((embedded, context), dim = 2)            
        output, hidden = self.rnn(emb_con, hidden)
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), 
                           dim = 1)
        prediction = self.fc_out(output)  
        return prediction, hidden


class Encoder_RNNSearch(nn.Module):
    def __init__(self, input_size, emb_size, enc_hidden_size, dec_hidden_size,  num_layers, device, dropout_proba):
        super().__init__()       
        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn = RNN(emb_size, enc_hidden_size,  num_layers, device, bidirectional = True)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)
        self.dropout = nn.Dropout(dropout_proba)
        
    def forward(self, source):
        embedded = self.dropout(self.embedding(source))  
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return outputs, hidden

class Decoder_RNNSearch(nn.Module):
    def __init__(self, output_size, emb_size, enc_hidden_size, dec_hidden_size, num_layers, device, dropout_proba, attention):
        super().__init__()

        self.output_size = output_size
        self.attention = attention   
        self.embedding = nn.Embedding(output_size, emb_size)
        self.rnn = RNN((enc_hidden_size * 2) + emb_size, dec_hidden_size, num_layers, device, bidirectional = True)
        self.fc_out = nn.Linear((enc_hidden_size * 2) + dec_hidden_size + emb_size, output_size) 
        self.dropout = nn.Dropout(dropout_proba)
        
    def forward(self, input, hidden, encoder_outputs):          
   
        input = input.unsqueeze(0) 
        embedded = self.dropout(self.embedding(input))     
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        assert (output == hidden).all() 
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0) 
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        return prediction, hidden.squeeze(0)

class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()      
        self.attn = nn.Linear((enc_hidden_size * 2) + dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1, bias = False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)
