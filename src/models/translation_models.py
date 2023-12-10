from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from metrics.losses import Loss
from models.decoder import Decoder
from models.encoder import Encoder

import datetime
import os

from utils.plotting import * 
from global_variables import DATA_DIR

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
        self.beam_search_flag = training_config.get("beam_search", False)
        if self.load_last_checkpoints:
            self.load_last_model()

        self.losses = []
        if not os.path.exists(DATA_DIR / f"outputs/"):
            os.makedirs(DATA_DIR / f"outputs/")


    def forward(self, x: torch.Tensor, validation: bool = False) -> torch.Tensor:
        # Forward pass through encoder and decoder
        encoder_output, _ = self.encoder(x)
        decoder_output, allignments= self.decoder(encoder_output)
        return (decoder_output, allignments) if validation else decoder_output

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
        
        directory = os.path.join(DATA_DIR, directory)

        # Create a directory with timestamp
        save_dir = os.path.join(directory, self.timestamp)
        os.makedirs(directory, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        # Save model
        save_path = os.path.join(save_dir, "model.pth")
        torch.save(self.state_dict(), save_path)
        print(f"Model saved at {save_path}")
        with open(save_dir + "/losses.txt", "w") as myfile:
            myfile.write("{}\n".format(self.timestamp))
            for loss in self.losses:
                myfile.write("{}\n".format(loss))

    def load_model(self, path: str) -> None:
        # Check compatibility
        try:
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
            print(f"Model loaded from {path}")
        except RuntimeError as e:
            print(f"Failed to load model from {path}: {str(e)}")
    
    def load_last_model(self, directory: str = "best_models/") -> None:
        directory = os.path.join(DATA_DIR, directory)
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
                self.losses.append(loss)
                if i % self.print_every == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss}")
                    #add losses to a text file
                if i % self.save_every == 0:
                    self.save_model( "checkpoints/")
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
            x_sentences, y_sentences = val_sample["english"]["sentences"], val_sample["french"]["sentences"]
            output, allignments= self.forward(x.to(self.device), validation=True)
            loss = self.criterion(output, y.to(self.device)) 
            total_loss += loss.item()
            if i == 0:
                prediction = output[:4]
                prediction[:,:,-2] = torch.min(prediction)
                prediction_idx = self.beam_search(prediction, 3) if self.beam_search_flag else torch.argmax(prediction, dim=-1)

                sample = self.sample_translation(x[:4], prediction_idx, y[:4])

                output_text = f"\tSource: {sample[0][0]}\n"
                output_text += f"\tPrediction: {sample[1][0]}\n"
                output_text += f"\tTranslation: {sample[2][0]}\n"
                
                with open(DATA_DIR / f"outputs/outputs_{self.timestamp}.txt", "w") as myfile:
                    myfile.write(output_text)
                print(output_text)
                self.plot_attention(sample[0], sample[1], allignments[:4])
        return total_loss / len(val_loader)

    @property
    def timestamp(self):
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def sample_translation(self, source, prediction, translation):
        source_sentences = []
        prediction_sentences = []
        translation_sentences = []
        for i in range(source.shape[0]):
            s, p, t = source[i], prediction[i], translation[i]
            # Sample a translation from the model
            source_sentences.append(self.idx_to_word(s, self.english_vocab))
            prediction_sentences.append(self.idx_to_word(p, self.french_vocab))
            translation_sentences.append(self.idx_to_word(t, self.french_vocab))

        return [source_sentences, prediction_sentences, translation_sentences]

    def idx_to_word(self, idx: torch.Tensor, vocab: List):
        # Convert index to word
        idx = idx.cpu().detach().numpy()
        phrase = " ".join(list(vocab[idx[(idx < len(vocab))]]))
        phrase = phrase.replace("  ", " ")
        return phrase

    def beam_search(self, tensor, beam_size):
        batch_size, len_seq, vocab_size = tensor.size()
        device = tensor.device
        # Initialize the beam search output tensor
        output = torch.zeros(batch_size, len_seq, dtype=torch.long, device=device)

        # Loop over each sequence in the batch
        for b in range(batch_size):
            # Initialize the beam search candidates
            candidates = [(torch.tensor([], dtype=torch.long, device=device), 0)]

            # Loop over each time step in the sequence
            for t in range(len_seq):
                # Get the scores for the next time step
                scores = F.log_softmax(tensor[b, t], dim=-1)

                # Generate new candidates by expanding the existing ones
                new_candidates = []
                for seq, score in candidates:
                    for i in range(vocab_size):
                        # Check if the new index is the same as the last index in the sequence
                        if len(seq) > 0 and i == seq[-1]:
                            continue

                        new_seq = torch.cat([seq, torch.tensor([i], dtype=torch.long, device=device)])
                        new_score = score + scores[i]
                        new_candidates.append((new_seq, new_score))

                # Select the top-k candidates based on the score
                new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

                # Update the candidates for the next time step
                candidates = new_candidates
            # Select the sequence with the highest score
            best_seq, _ = max(candidates, key=lambda x: x[1])

            # Store the best sequence in the output tensor
            output[b] = best_seq

        return output

    def plot_attention(self, source, prediction, allignments):
        source_list = [s.split(" ") for s in source]
        prediction_list = [p.split(" ") for p in prediction]
        alls = []
        alignments = allignments.cpu().detach().numpy()
        for i in range(alignments.shape[0]):
            alls.append(alignments[i, :len(prediction_list[i]), :len(source_list[i])])


        data = {
            f"phrase {i}": (source_list[i], prediction_list[i], alls[i]) for i in range(len(source_list))
        }
        plot_alignment(data, save_path=f"alignment_{self.timestamp}.png")