from torch import nn
from models.encoder import Encoder
from models.decoder import Decoder
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


eng_tensor = []
fr_tensor = []

# Concatenate tensors to create a single tensor
concat_tensor = torch.cat((eng_tensor, fr_tensor), dim=1)

# Split the concatenated tensor into training and test sets
train_size = int(0.8 * len(concat_tensor))  
test_size = len(concat_tensor) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(concat_tensor, [train_size, test_size])

# Create DataLoaders
batch_size = 64 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class AlignAndTranslate(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(AlignAndTranslate, self).__init__()
        self.encoder = Encoder(encoder_params)
        self.decoder = Decoder(decoder_params)

    def forward(self, x):
        encoder_output, encoder_hidden = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

    def train_step(self, input_tensor, target_tensor, optimizer, criterion):
        optimizer.zero_grad()

        # Encoder
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)

        # Initialize decoder input with SOS_token
        decoder_input = torch.tensor([['startofsequencetoken']])  
        decoder_hidden = encoder_hidden

        # Initialize loss
        loss = 0

        # Teacher forcing: Feed the target as the next input
        for di in range(target_tensor.size(0)):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Convert probabilities to indices
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])

            if decoder_input.item() == 'end of sequence indice':  
                break

        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, train_loader, n_epochs):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)  
        criterion = nn.CrossEntropyLoss() 

        for epoch in range(n_epochs):
            total_loss = 0

            for batch in train_loader:
                input_data, target_data = batch[:, :eng_tensor.size(1)], batch[:, eng_tensor.size(1):]
                loss = self.train_step(input_data, target_data, optimizer, criterion)
                total_loss += loss

            average_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {average_loss}')

    def predict(self, test_loader, max_length=max_length):
        self.eval()  # Set the model to evaluation mode
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                input_data, _ = batch
                encoder_outputs, encoder_hidden = self.encoder(input_data)

                # Initialize decoder input with SOS_token
                decoder_input = torch.tensor([['startofsequencetoken']]) 
                decoder_hidden = encoder_hidden
                decoded_words = []

                # Generate translation one word at a time
                for di in range(max_length):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    
                    # Convert probabilities to indices
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    if decoder_input.item() == 'end of sequence indice':  
                        break
                    else:
                        decoded_words.append(decoder_input.item())

                predictions.append(decoded_words)

        return predictions



