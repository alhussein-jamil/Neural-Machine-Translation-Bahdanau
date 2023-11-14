import torch.nn as nn
import rnn
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




    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)
