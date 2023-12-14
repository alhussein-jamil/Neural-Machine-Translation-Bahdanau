from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from metrics.losses import Loss
from metrics import bleu
from models.decoder import Decoder
from models.encoder import Encoder

import datetime
import os

from utils.plotting import *
from global_variables import DATA_DIR

from sacremoses import MosesDetokenizer


class AlignAndTranslate(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        # Initialize encoder and decoder
        self.encoder = Encoder(**kwargs.get("encoder", {}))
        self.decoder = Decoder(**kwargs.get("decoder", {}))

        # Training configuration
        training_config = kwargs.get("training", {})
        self.criterion = training_config.get(
            "criterion", Loss(nn.NLLLoss())
        )
        self.optimizer = training_config.get(
            "optimizer", torch.optim.Adam(self.parameters())
        )
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
        self.start_time = self.timestamp

        self.train_losses = []
        self.val_losses = [1e10]
        self.local_dir = DATA_DIR / ("trained_models/" + self.start_time)
        self.models_dir = self.local_dir / "checkpoints/"
        self.best_models_dir = self.local_dir / "best_models/"
        self.output_dir = self.local_dir / "outputs/"
        self.plot_dir = self.local_dir / "plots/"
        self.bleu_scores = [0.0]
        os.makedirs(DATA_DIR / "trained_models/", exist_ok=True)
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.best_models_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        if self.load_last_checkpoints:
            self.load_last_model()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through encoder and decoder
        encoder_output, _ = self.encoder(x)
        decoder_output, allignments = self.decoder(encoder_output)
        return (decoder_output, allignments)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        # Training step
        self.optimizer.zero_grad()

        output, allignments = self.forward(x.to(self.device))

        loss = self.criterion(output, y) / output.shape[-1]
        loss *= (1e4)/4.0

        truncated_loss = loss.clone()
        # normalize L2 loss so that it stays under 1
        if torch.norm(loss) > 1:
            truncated_loss = loss / torch.norm(loss)
        truncated_loss.backward()
        self.optimizer.step()
        return loss.item(), output, allignments

    def save_model(self, best: bool = False) -> None:
        # Create a directory with timestamp
        save_dir = os.path.join(
            self.models_dir if not best else self.best_models_dir, self.timestamp
        )
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

    def load_last_model(self) -> None:
        # Load last model
        try:
            if not os.listdir(self.best_models_dir):
                print(f"No model found in {self.best_models_dir}")
                return
            last_model = sorted(os.listdir(self.best_models_dir))[-1]
            self.load_model(os.path.join(self.best_models_dir, last_model))
        except IndexError:
            print(f"No model found in {self.best_models_dir}")

    def train(self, train_loader, val_loader) -> None:
        # Training loop
        for epoch in range(self.epochs):
            losses = []
            for i, train_sample in enumerate(train_loader):
                x, y = (
                    train_sample["english"]["idx"],
                    train_sample["french"]["idx"],
                )
                x = x.to(self.device)
                y = y.to(self.device)
                loss, output, allignments = self.train_step(x, y)
                losses.append(loss)

                if i % self.print_every == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss}")
                    # add losses to a text file
                if i % self.save_every == 0:
                    self.save_model()

            with torch.no_grad():
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                self.display(output, allignments, x, y, val=False)
                print(f"Epoch: {epoch}, Validation Loss: {val_loss}")
            self.train_losses.append(sum(losses) / len(losses))
            new = True
            if os.path.exists(self.output_dir / "losses.txt"):
                new = False

            with open(
                self.output_dir / "losses.txt",
                "a" if not new else "w",
                encoding="utf-8",
            ) as myfile:
                myfile.write(
                    "{} {} {}\n".format(self.train_losses[-1], self.val_losses[-1], torch.mean(self.bleu_scores[-1]).float())   
                )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(best=True)

    def display(
        self,
        output: torch.tensor,
        allignments: torch.tensor,
        x: torch.tensor,
        y: torch.tensor,
        val: bool = True,
    ):
        prediction = output[:4]
        prediction[:, :, -2] = torch.min(
            prediction
        )  # set the <unk> token to the minimum value so that it is not selected
        prediction_idx = (
            self.beam_search(prediction, 5)
            if self.beam_search_flag
            else torch.argmax(prediction, dim=-1)
        )
        self.bleu_scores.append(
            bleu(
                y.cpu().detach().numpy(),
                prediction_idx.cpu().detach().numpy(),
                n= 4,
            )
        )

        sample = self.sample_translation(x[:4], prediction_idx, y[:4])
        name = "Validation" if val else "Training"
        translations = f"{name} samples:\n"
        for s in range(4):
            translations += f"\tSource: {sample[0][s]}\n"
            translations += f"\tPrediction: {sample[1][s]}\n"
            translations += f"\tTranslation: {sample[2][s]}\n"
            translations += "\n"
        with open(
            self.output_dir / (self.timestamp + ".txt"), "a", encoding="utf-8"
        ) as myfile:
            myfile.write(translations)
        print(translations)
        if val:
            bleu_scores = self.bleu_scores[-1][:4]
            self.plot_attention(sample[0], sample[1], allignments[:4], bleu_scores)

    def evaluate(self, val_loader) -> float:
        # Evaluation function
        total_loss = 0
        for i, val_sample in enumerate(val_loader):
            x, y = val_sample["english"]["idx"], val_sample["french"]["idx"]
            output, allignments = self.forward(x.to(self.device))
            loss = self.criterion(output, y.to(self.device))
            total_loss += loss.item()
            if i == 0:
                self.display(output, allignments, x, y, val=True)

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
            source_sentences.append(
                self.idx_to_word(s, self.english_vocab, language="en")
            )
            translation_sentences.append(
                self.idx_to_word(t, self.french_vocab, language="fr")
            )
            prediction_sentences.append(
                self.idx_to_word(p, self.french_vocab, language="fr")
            )
            # only keep the first len(tranlation_sentences[i]) tokens
            prediction_sentences[i] = " ".join(
                prediction_sentences[i].split(" ")[
                    : len(translation_sentences[i].split(" "))
                ]
            )

        return [source_sentences, prediction_sentences, translation_sentences]

    def idx_to_word(self, idx: torch.Tensor, vocab: List, language="fr") -> str:
        # Convert index to word
        idx = idx.cpu().detach().numpy()
        tokens = list(vocab[idx[(idx < len(vocab))]])
        detokenizer = MosesDetokenizer(lang=language)
        phrase = detokenizer.detokenize(tokens, return_str=True)
        phrase = phrase.replace("  ", " ")
        return phrase

    def beam_search(self, tensor, beam_size):
        batch_size, len_seq, _ = tensor.size()
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
                    # Calculate the cumulative scores for all words in the candidate sequence
                    cumulative_scores = score + scores

                    # Find the top-k candidates based on the cumulative scores
                    top_k_indices = torch.topk(cumulative_scores, beam_size).indices
                    for i in top_k_indices:
                        # avoid repitition
                        repition = False
                        # check repitions
                        for length in range(len(seq)):
                            concatenated = torch.cat(
                                [
                                    seq[len(seq) - length :],
                                    torch.tensor([i], dtype=torch.int64, device=device),
                                ]
                            )
                            previous = seq[
                                len(seq) - 2 * length - 1 : len(seq) - length
                            ]
                            if len(previous) == len(concatenated):
                                repition = (concatenated == previous).all()
                            if repition:
                                break

                        if repition:
                            continue

                        new_seq = torch.cat(
                            [seq, torch.tensor([i], dtype=torch.long, device=device)]
                        )
                        new_score = cumulative_scores[i]
                        new_candidates.append((new_seq, new_score))

                # Select the top-k candidates from all expanded candidates
                new_candidates = sorted(
                    new_candidates, key=lambda x: x[1], reverse=True
                )[:beam_size]

                # Update the candidates for the next time step
                candidates = new_candidates

            # Select the sequence with the highest score
            best_seq, _ = max(candidates, key=lambda x: x[1])

            # Store the best sequence in the output tensor
            output[b] = best_seq

        return output

    def plot_attention(self, source, prediction, allignments, titles):
        source_list = [s.split(" ") for s in source]
        prediction_list = [p.split(" ") for p in prediction]
        alls = []
        alignments = allignments.cpu().detach().numpy()
        for i in range(alignments.shape[0]):
            alls.append(alignments[i, : len(prediction_list[i]), : len(source_list[i])])

        data = {
            f"phrase {i}: bleu-score of {titles[i]*100.0}": (source_list[i], prediction_list[i], alls[i])
            for i in range(len(source_list))
        }
        plot_alignment(data, save_path=self.plot_dir / (self.timestamp + ".png"))
