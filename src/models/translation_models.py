from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from metrics.losses import Loss
from models.decoder import Decoder
from models.encoder import Encoder

# class Loss(nn.Module):
#     def __init__(self, loss_fn) -> None:
#         super().__init__()
#         self.loss_fn = loss_fn

#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """
#             x of shape (batch_size, sequence_length, vocab_size)
#             y of shape (batch_size, sequence_length)
#         """

#         y = y.long()

#         y_idx = F.one_hot(y, num_classes=x.shape[-1]).float()

#         losses = self.loss_fn(x, y_idx)

#         return losses.mean()


class AlignAndTranslate(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = Encoder(**kwargs.get("encoder", {}))
        self.decoder = Decoder(**kwargs.get("decoder", {}))

        training_config = kwargs.get("training", {})

        self.criterion = training_config.get("criterion", Loss(nn.CrossEntropyLoss(reduction="sum")))
        self.optimizer = training_config.get("optimizer", torch.optim.Adam(self.parameters(), lr=1e-4))
        self.device = training_config.get("device", "cpu")
        self.epochs = training_config.get("epochs", 100)
        self.batch_size = training_config.get("batch_size", 32)
        self.print_every = training_config.get("print_every", 100)
        self.save_every = training_config.get("save_every", 1000)
        self.checkpoint = training_config.get("checkpoint", "checkpoint.pth")
        self.best_model = training_config.get("best_model", "best_model.pth")
        self.output_vocab_size = training_config.get("output_vocab_size", 100)
        self.best_val_loss = float("inf")
        self.english_vocab = training_config.get("english_vocab", [])
        self.french_vocab = training_config.get("french_vocab", [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_output, _ = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        output = self.forward(x.to(self.device))
        loss = self.criterion(output, y) / self.batch_size
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        print(f"Model saved at {path}")

    def train(self, train_loader, val_loader) -> None:
        for epoch in range(self.epochs):
            for i, train_sample in enumerate(train_loader):
                x, y = (
                    train_sample["english"]["idx"],
                    train_sample["french"]["idx"],
                )
                x = x.to(self.device)
                y = y.to(self.device)
                loss = self.train_step(x, y)
                if i % self.print_every == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss}")
                if i % self.save_every == 0:
                    self.save_model(self.checkpoint)
            with torch.no_grad():
                val_loss = self.evaluate(val_loader)

            for i, val_sample in enumerate(val_loader):
                x, y = val_sample["english"]["idx"], val_sample["french"]["idx"]
                prediction = self.forward(x[0].unsqueeze(0).to(self.device)).squeeze(0)
                prediction_idx = torch.argmax(prediction, dim=1)
                print(prediction_idx)
                sample = self.sample_translation(x[0], prediction_idx, y[0])
                break
            print(f"Source: {sample[0]}")
            print(f"Prediction: {sample[1]}")
            print(f"Translation: {sample[2]}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(self.best_model)

    def evaluate(self, val_loader) -> float:
        total_loss = 0
        for i, val_sample in enumerate(val_loader):
            x, y = val_sample["english"]["idx"], val_sample["french"]["idx"]
            output = self.forward(x.to(self.device))
            loss = self.criterion(output, y.to(self.device)) / self.batch_size
            total_loss += loss.item()
        return total_loss / len(val_loader)

    def sample_translation(self, source, prediction, translation):
        """
        Sample a translation from the model.
        given the indicies of the prediction and the vocab, return the sentence

        """
        source_sentence = self.idx_to_word(source, self.english_vocab)
        prediction_sentence = self.idx_to_word(prediction, self.french_vocab)
        translation_sentence = self.idx_to_word(translation, self.french_vocab)

        return source_sentence, prediction_sentence, translation_sentence

    def idx_to_word(self, idx: torch.Tensor, vocab: List):
        idx = idx.cpu().detach().numpy()
        phrase = " ".join(list(vocab[idx[(idx < len(vocab))]]))
        phrase = phrase.replace("  ", " ")
        return phrase
