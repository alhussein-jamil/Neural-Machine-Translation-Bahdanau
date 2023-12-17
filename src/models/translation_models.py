from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from metrics.losses import Loss
from metrics import bleu_seq
from models.decoder import Decoder
from models.encoder import Encoder

import datetime
import os

from utils.plotting import *
from global_variables import DATA_DIR

from sacremoses import MosesDetokenizer, MosesTokenizer
import math

from data_preprocessing import *

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
            # "optimizer", torch.optim.Adadelta(self.parameters(), eps=1e-6, rho = 0.95)
            "optimizer", torch.optim.Adam(self.parameters(), amsgrad=True,lr=1e-4)
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
        self.Tx = training_config["Tx"]
        self.Ty = training_config["Ty"]

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

    def calc_loss(self, output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        loss = self.criterion(output, y) 

        L2_reg = torch.tensor(0.0).to(self.device)

        for param in self.parameters():
            L2_reg += torch.norm(param)
             
        L2_reg*=3e-5

        loss += L2_reg

        return loss

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        # Training step
        self.optimizer.zero_grad()

        output, allignments = self.forward(x.to(self.device))

        loss = self.calc_loss(output, y.to(self.device))

        loss.backward()

        #Gradient Value Clipping
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

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
            state_dict = torch.load(path, map_location=self.device)
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
        train_file_exists = os.path.exists(self.output_dir / "train_losses.txt")
        val_file_exists = os.path.exists(self.output_dir / "val_losses.txt")
        # Training loop
        for epoch in range(math.ceil(self.epochs)):
            losses = []
            for i, train_sample in enumerate(train_loader):
                exact_epoch = epoch + i / len(train_loader)
                if exact_epoch > self.epochs:
                    break
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

                with open(
                    self.output_dir / "train_losses.txt",
                    "a" if not train_file_exists else "w",
                ) as myfile:
                    myfile.write(f"{loss}\n")
            val_losses = []
            with torch.no_grad():
                val_loss = self.evaluate(val_loader)
                val_losses.append(val_loss)
                self.val_losses.append(val_loss)
                self.display(output, allignments, x, y, val=False)
                print(f"Epoch: {epoch}, Validation Loss: {val_loss}")
                with open(
                    self.output_dir / "val_losses.txt",
                    "a" if not val_file_exists else "w",
                ) as myfile:
                    myfile.write(f"{val_loss}\n")
            self.train_losses.append(sum(losses) / len(losses))

            losses_exist = os.path.exists(self.output_dir / "losses.txt")
                
            with open(
                self.output_dir / "losses.txt",
                "a" if not losses_exist else "w",
                encoding="utf-8",
            ) as myfile:
                myfile.write(
                    "{} {} {}\n".format(self.train_losses[-1], self.val_losses[-1], torch.mean(self.bleu_scores[-1]).float())   
                )

            if sum(val_losses) / len(val_losses) < self.best_val_loss:
                self.best_val_loss = sum(val_losses) / len(val_losses)
                self.save_model(best=True)

    def display(
        self,
        output: torch.tensor,
        allignments: torch.tensor,
        x: torch.tensor,
        y: torch.tensor,
        val: bool = True,
    ):
        random_idx = torch.randint(0, len(x), (4,)) 
        prediction = output[random_idx]
        prediction[:, :, -2] = torch.min(
           prediction
        )  # set the <unk> token to the minimum value so that it is not selected
        prediction_idx = (
            self.beam_search(prediction, 5)
            if self.beam_search_flag
            else torch.argmax(prediction, dim=-1)
        )


        sample = self.sample_translation(x[random_idx], prediction_idx, y[random_idx])

        self.bleu_scores.append(
            bleu_seq(sample[1], sample[2], n=4)
        )


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
        bleu_scores = self.bleu_scores[-1]
        self.plot_attention(sample[0], sample[1], allignments[:4], bleu_scores, val=val)

    def evaluate(self, val_loader) -> float:
        # Evaluation function
        total_loss = 0
        for i, val_sample in enumerate(val_loader):
            x, y = val_sample["english"]["idx"], val_sample["french"]["idx"]
            output, allignments = self.forward(x.to(self.device))
            loss = self.calc_loss(output, y.to(self.device))
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
                    : len(translation_sentences[i].split(" ")) + 3
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

    def beam_search(self, tensor, beam_size, threshold=-2.8):
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
                        if scores[i] < threshold:
                            #add padding 
                            new_seq = torch.cat(
                            [seq, torch.tensor([len(self.french_vocab)], dtype=torch.long, device=device)]
                        )
                        else: 
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

    def plot_attention(self, source, prediction, allignments, titles, val=True, path=None):
        source_list = [s.split(" ") for s in source]
        prediction_list = [p.split(" ") for p in prediction]
        alls = []
        alignments = allignments.cpu().detach().numpy()
        for i in range(alignments.shape[0]):
            alls.append(alignments[i, : len(prediction_list[i]), : len(source_list[i])])

        data = {
            f"phrase {i}: bleu-score of {int(titles[i]*100.0)}%": (source_list[i], prediction_list[i], alls[i])
            for i in range(len(source_list))
        }
        return plot_alignment(data, save_path=self.plot_dir / (self.timestamp+"_{}".format("val" if val else "train")+ ".png") if path is None else None)

    def eval(self, dataloader, max_len):
        #try sentences of length till Tx
        bleu_scores = torch.zeros(max_len,len(dataloader))
        original_sentences = []
        predicted_sentences = []
        for i, val_sample in enumerate(dataloader):
            x, y = val_sample["english"]["idx"], val_sample["french"]["idx"]
            output, _ = self.forward(x.to(self.device))
            prediction_idx = torch.argmax(output, dim=-1)
            for length in range(1, max_len):
                translation = self.sample_translation(x[:length], prediction_idx[:length], y[:length])
                original_sentences += translation[2]
                predicted_sentences += translation[1]

        bleu_scores = bleu_seq(original_sentences, predicted_sentences, n=4).reshape(max_len,-1)
        return torch.mean(bleu_scores, dim=1)


    def translate_sentence(self, sentences: List[Dict[Any,Any]]):

        tokenizer_en = MosesTokenizer(lang="en")
        tokenizer_fr = MosesTokenizer(lang="fr")
        tokenizer = TokenizerWrapper(tokenizer_en, tokenizer_fr)
        to_id = toIdTransform(self.english_vocab, self.french_vocab, torch)

        # Initialize tensors for train and validation data
        idx_tensor_en = torch.zeros((len(sentences), self.Tx), dtype=torch.int16)
        idx_tensor_fr = torch.zeros((len(sentences), self.Ty), dtype=torch.int16)
        treated_sentences = []
        for sentence in sentences:
            sentence = tokenizer.tokenize_function(sentence)
            sentence = to_id(sentence)  
            treated_sentences.append(sentence)
        
        pad_multiprocess(treated_sentences, idx_tensor_en, idx_tensor_fr, self.Tx, self.Ty, len(self.english_vocab), len(self.french_vocab), False)

        output, alignment = self.forward(idx_tensor_en.to(self.device))
    
        output[:, :, -2] = torch.min(
            output
        )  # set the <unk> token to the minimum value so that it is not selected
        prediction_idx = (
            self.beam_search(output,3,-2.8) if self.beam_search_flag else self.greedy_search_batch(output)
            
        )

        sample = self.sample_translation(idx_tensor_en, prediction_idx, idx_tensor_fr)

        return sample, alignment


    def greedy_search_batch(self, tensors, avoid_repetition=True):
        batch_size, len_seq, vocab_size = tensors.size()
        output = torch.zeros(batch_size, len_seq, dtype=torch.long)
        softmaxed_ouput = F.softmax(tensors*1.2, dim=-1)

        for b in range(batch_size):
            tensor = tensors[b]
            for i in range(len_seq):
                if avoid_repetition:
                    # pick an index with probability propotional to the softmaxed ouput 
                    found = False
                    while not found:
                        idx = torch.multinomial(softmaxed_ouput[b,i], 1).item()
                        repetition = False
                        # check repetitions
                        for length in range(i):
                            concatenated = torch.cat(
                                [
                                    output[b, i - length :i],
                                    torch.tensor([idx], dtype=torch.int64),
                                ]
                            )
                            previous = output[
                                b, i - 2 * length - 1 : i - length
                            ]
                            if len(previous) == len(concatenated):
                                repetition = (concatenated == previous).all()
                            if repetition:
                                break
                        if not repetition:
                            output[b, i] = idx
                            found = True 
                            break
                else:
                    # Simply get the index of the maximum value in tensor[i]
                    output[b, i] = torch.argmax(tensor[i])

        return output