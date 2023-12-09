from torch import nn
from models.encoder import Encoder
from models.decoder import Decoder
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch 




class AlignAndTranslate(nn.Module):
    def __init__(self, encoder_params, decoder_params, optimizer_params=None, criterion=None):
        super(AlignAndTranslate, self).__init__()
        self.encoder = Encoder(**encoder_params)
        self.decoder = Decoder(**decoder_params)
        
        # Initialize optimizer and criterion
        if optimizer_params is not None:
            self.optimizer = optim.SGD(self.parameters(), **optimizer_params)
        else:
            self.optimizer = None
        if criterion is not None :
            self.criterion =  nn.CrossEntropyLoss() 
        else :
            self.criterion = None

    def forward(self, x):
        encoder_output, _ = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

    def train_step(self, input_data, target_data):
        optimizer = self.optimizer
        criterion = self.criterion

        if optimizer is None or criterion is None:
            raise ValueError("Optimizer and criterion must be provided for training.")

        optimizer.zero_grad()

        # Pass through the model
        output = self(input_data)
        # Compute the loss
        loss = criterion(output, target_data)
        loss.backward()
        # Optimization step
        optimizer.step()

        return loss.item()

    def train(self, train_loader, n_epochs):
        if self.optimizer is None or self.criterion is None:
            raise ValueError("Optimizer and criterion must be provided for training.")

        for epoch in range(n_epochs):
            total_loss = 0

            # Training
            self.train()
            for batch in train_loader:
                input_data, target_data = batch["english"]["idx"], batch["french"]["idx"]
                loss = self.train_step(input_data, target_data)
                total_loss += loss

            average_loss = total_loss / len(train_loader)
            print(f'Training - Epoch {epoch + 1}/{n_epochs}, Loss: {average_loss}')

    def predict(self, test_loader):
        self.eval()  # Mettre le modèle en mode évaluation
        predictions = []
        val_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                input_data, target_data = batch["english"]["idx"], batch["french"]["idx"]
                output = self(input_data)

                # Convertir les probabilités en indices
                _, topi = output.topk(1)
                predictions.append(topi.squeeze().detach())  # Détacher sans convertir en numpy array
                val_loss += self.criterion(output, target_data).item()

            average_val_loss = val_loss / len(test_loader)
            print(f'Validation Loss: {average_val_loss}')

        # Concaténer les prédictions pour obtenir un tenseur
        predictions_tensor = torch.cat(predictions, dim=1)

        return predictions_tensor

