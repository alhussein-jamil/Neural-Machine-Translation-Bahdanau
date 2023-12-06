from torch import nn
from models.encoder import Encoder
from models.decoder import Decoder
import torch

class AlignAndTranslate(nn.Module):

    def _init_(self, *args, **kwargs) -> None:

        super()._init_(*args, **kwargs)

        self.encoder = Encoder(kwargs.get("encoder", {}))
        self.decoder = Decoder(kwargs.get("decoder", {}))

        training_config = kwargs.get("training", {})

        self.criterion = training_config.get("criterion", nn.CrossEntropyLoss())
        self.optimizer = training_config.get("optimizer", torch.optim.Adam(self.parameters()))
        self.device = training_config.get("device", "cpu")
        self.epochs = training_config.get("epochs", 10)
        self.batch_size = training_config.get("batch_size", 32)
        self.print_every = training_config.get("print_every", 100)
        self.save_every = training_config.get("save_every", 1000)
        self.checkpoint = training_config.get("checkpoint", "checkpoint.pth")
        self.best_model = training_config.get("best_model", "best_model.pth")


    def forward(self, x):
        encoder_output, _ = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output
    

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            self.train()
            for i, train_sample in enumerate(train_loader):
                x, y = train_sample["english"]["idx"], train_sample["french"]["idx"]
                loss = self.train_step(x, y)
                if i % self.print_every == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss}")
                if i % self.save_every == 0:
                    torch.save(self.state_dict(), self.checkpoint)
                    print(f"Checkpoint saved at {self.checkpoint}")
            val_loss = self.evaluate(val_loader)
            print(f"Epoch: {epoch}, Validation Loss: {val_loss}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.state_dict(), self.best_model)
                print(f"Best model saved at {self.best_model}")