from torch import nn
from models.encoder import Encoder
from models.decoder import Decoder
import torch
<<<<<<< HEAD
<<<<<<< HEAD
import torch.nn.functional as F

class AlignAndTranslate(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = Encoder(**kwargs.get("encoder", {}))
        self.decoder = Decoder(**kwargs.get("decoder", {}))
=======

class AlignAndTranslate(nn.Module):

    def _init_(self, *args, **kwargs) -> None:

        super()._init_(*args, **kwargs)

        self.encoder = Encoder(kwargs.get("encoder", {}))
        self.decoder = Decoder(kwargs.get("decoder", {}))
>>>>>>> ecdb713 (Refactor code and update dependencies)
=======
import torch.nn.functional as F

class AlignAndTranslate(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = Encoder(**kwargs.get("encoder", {}))
        self.decoder = Decoder(**kwargs.get("decoder", {}))
>>>>>>> 4addc1d (Fix alignment vector computation in Decoder)

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
<<<<<<< HEAD
<<<<<<< HEAD
        self.output_vocab_size = training_config.get("output_vocab_size", 100)
        self.best_val_loss = float("inf")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_output, _ = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        output = self.forward(x.to(self.device))
        y_idx = F.one_hot(y.to(self.device).long(), num_classes=self.output_vocab_size).float()
        loss = self.criterion(output, y_idx)
=======
=======
        self.output_vocab_size = training_config.get("output_vocab_size", 100)
        self.best_val_loss = float('inf')
>>>>>>> 4addc1d (Fix alignment vector computation in Decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_output, _ = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.optimizer.zero_grad()
<<<<<<< HEAD
        output = self.forward(x)
        loss = self.criterion(output, y)
>>>>>>> ecdb713 (Refactor code and update dependencies)
=======
        output = self.forward(x.to(self.device))
        y_idx = F.one_hot(y.to(self.device).long(), num_classes=self.output_vocab_size).float()
        loss = self.criterion(output, y_idx)
>>>>>>> 4addc1d (Fix alignment vector computation in Decoder)
        loss.backward()
        self.optimizer.step()
        return loss.item()

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 4addc1d (Fix alignment vector computation in Decoder)
    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        print(f"Model saved at {path}")

<<<<<<< HEAD

    def train(self, train_loader, val_loader) -> None:
        for epoch in range(self.epochs):
=======
    def train(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            self.train()
>>>>>>> ecdb713 (Refactor code and update dependencies)
=======
    @property
    def best_val_loss(self) -> float:
        return self._best_val_loss

    def train(self, train_loader, val_loader) -> None:
        for epoch in range(self.epochs):
>>>>>>> 4addc1d (Fix alignment vector computation in Decoder)
            for i, train_sample in enumerate(train_loader):
                x, y = train_sample["english"]["idx"], train_sample["french"]["idx"]
                loss = self.train_step(x, y)
                if i % self.print_every == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss}")
                if i % self.save_every == 0:
<<<<<<< HEAD
<<<<<<< HEAD
                    self.save_model(self.checkpoint)
            with torch.no_grad():
                val_loss = self.evaluate(val_loader)
            print(f"Epoch: {epoch}, Validation Loss: {val_loss}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(self.best_model)

    def evaluate(self, val_loader) -> float:
        total_loss = 0
        for i, val_sample in enumerate(val_loader):
            x, y = val_sample["english"]["idx"], val_sample["french"]["idx"]
            output = self.forward(x.to(self.device))
            y_idx = F.one_hot(y.to(self.device).long(), num_classes=self.output_vocab_size).float()
            loss = self.criterion(output, y_idx)
            total_loss += loss.item()
        return total_loss / len(val_loader)
    
=======
                    torch.save(self.state_dict(), self.checkpoint)
                    print(f"Checkpoint saved at {self.checkpoint}")
            val_loss = self.evaluate(val_loader)
            print(f"Epoch: {epoch}, Validation Loss: {val_loss}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.state_dict(), self.best_model)
                print(f"Best model saved at {self.best_model}")
>>>>>>> ecdb713 (Refactor code and update dependencies)
=======
                    self.save_model(self.checkpoint)
            with torch.no_grad():
                val_loss = self.evaluate(val_loader)
            print(f"Epoch: {epoch}, Validation Loss: {val_loss}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(self.best_model)
>>>>>>> 4addc1d (Fix alignment vector computation in Decoder)
