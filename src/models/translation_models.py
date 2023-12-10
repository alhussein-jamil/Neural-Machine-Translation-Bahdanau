from typing import List

import torch
from torch import nn

from metrics.losses import Loss
from models.decoder import Decoder
from models.encoder import Encoder

import datetime
import os

from utils.plotting import * 

class AlignAndTranslate(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        # Initialize encoder and decoder
        self.encoder = Encoder(**kwargs.get("encoder", {}))
        self.decoder = Decoder(**kwargs.get("decoder", {}))

        # Training configuration
        training_config = kwargs.get("training", {})
        self.criterion = training_config.get("criterion", Loss(nn.NLLLoss(reduction="mean")))
        self.optimizer = training_config.get("optimizer", torch.optim.Adam(self.parameters()))
        self.device = training_config.get("device", "cpu")
        self.epochs = training_config.get("epochs", 100)
        self.print_every = training_config.get("print_every", 100)
        self.save_every = training_config.get("save_every", 1000)
        self.output_vocab_size = training_config.get("output_vocab_size", 100)
        self.best_val_loss = float("inf")
        self.english_vocab = training_config.get("english_vocab", [])
        self.french_vocab = training_config.get("french_vocab", [])
        self.load_last_checkpoints = training_config.get("load_last_model", False)
        if self.load_last_checkpoints:
            self.load_last_model()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through encoder and decoder
        encoder_output, _ = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        # Training step
        self.optimizer.zero_grad()
        output = self.forward(x.to(self.device))
        loss = self.criterion(output, y)
        # normalize L2 loss so that it stays under 1
        # if torch.norm(loss) > 1:
        #     loss = loss / torch.norm(loss)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_model(self, directory: str = "checkpoints/") -> None:
        # Create a directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(directory, timestamp)
        os.makedirs(directory, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        # Save model
        save_path = os.path.join(save_dir, "model.pth")
        torch.save(self.state_dict(), save_path)
        print(f"Model saved at {save_path}")

    def load_model(self, path: str) -> None:
        # Check compatibility
        try:
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
            print(f"Model loaded from {path}")
        except RuntimeError as e:
            print(f"Failed to load model from {path}: {str(e)}")
    
    def load_last_model(self, directory: str = "best_models/") -> None:
        # Load last model
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
            if not os.listdir(directory):
                print(f"No model found in {directory}")
                return
            last_model = sorted(os.listdir(directory))[-1]
            self.load_model(os.path.join(directory, last_model, "model.pth"))
        except IndexError:
            print(f"No model found in {directory}")

    def train(self, train_loader, val_loader) -> None:
        # Training loop
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
                    self.save_model("checkpoints/")
            with torch.no_grad():
                val_loss = self.evaluate(val_loader)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model("best_models/")

    def evaluate(self, val_loader) -> float:
        # Evaluation function
        total_loss = 0
        for i, val_sample in enumerate(val_loader):
            x, y = val_sample["english"]["idx"], val_sample["french"]["idx"]
            output = self.forward(x.to(self.device))
            loss = self.criterion(output, y.to(self.device)) 
            total_loss += loss.item()
            if i == 0:
                prediction = output[0]
                prediction[:, -1] = -float("inf")  # Prevent the model from predicting the unknown token
                prediction_idx = self.beam_search_decoder(prediction, 5)
                sample = self.sample_translation(x[0], prediction_idx, y[0])
                print(f"Source: {sample[0]}")
                print(f"Prediction: {sample[1]}")
                print(f"Translation: {sample[2]}")

        return total_loss / len(val_loader)

    def sample_translation(self, source, prediction, translation):
        # Sample a translation from the model
        source_sentence = self.idx_to_word(source, self.english_vocab)
        prediction_sentence = self.idx_to_word(prediction, self.french_vocab)
        translation_sentence = self.idx_to_word(translation, self.french_vocab)

        return source_sentence, prediction_sentence, translation_sentence

    def idx_to_word(self, idx: torch.Tensor, vocab: List):
        # Convert index to word
        idx = idx.cpu().detach().numpy()
        phrase = " ".join(list(vocab[idx[(idx < len(vocab))]]))
        phrase = phrase.replace("  ", " ")
        return phrase

    def beam_search_decoder(self, data, k):
        # Beam search decoder
        sequences = [[list(), 0.0]]
        for row in data:
            all_candidates = list()
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score - torch.log(row[j])]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:k]
        return torch.tensor(sequences[0][0], dtype=torch.long)

    def plot(self, source):
        encoder_output, _ = self.encoder(source)

        allignments = self.decoder.alignment.forward_unoptimized