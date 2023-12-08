import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from sacremoses import MosesDetokenizer, MosesTokenizer
from torch.utils.data import Dataset


class TokenizerWrapper:
    """
    Wrapper class for tokenization using MosesTokenizer for English and French.
    """

    def __init__(self, tokenizer_en, tokenizer_fr):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr

    def preprocess_text(self, text):
        """
        Preprocess the input text, e.g., convert to lowercase.
        """
        return text.lower()

    def tokenize_function(self, examples):
        """
        Tokenize English and French translations in the examples.
        """
        preprocessed_en = self.preprocess_text(examples["translation"]["en"])
        preprocessed_fr = self.preprocess_text(examples["translation"]["fr"])

        return {
            "tokenized_en": self.tokenizer_en.tokenize(preprocessed_en),
            "tokenized_fr": self.tokenizer_fr.tokenize(preprocessed_fr),
        }


class toIdTransform:
    """
    Transform class to convert tokenized sentences to corresponding word IDs.
    """

    def __init__(self, most_frequent_words_en, most_frequent_words_fr, tor):
        self.most_frequent_words_en = most_frequent_words_en
        self.most_frequent_words_fr = most_frequent_words_fr
        self.word_to_id_en = {word: i for i, word in enumerate(most_frequent_words_en)}
        self.word_to_id_fr = {word: i for i, word in enumerate(most_frequent_words_fr)}
        self.tor = tor

    def __call__(self, tokenized):
        """
        Convert tokenized sentences to word IDs.
        """
        return {
            "ids_en": [
                self.word_to_id_en.get(token, len(self.most_frequent_words_en) - 1)
                for token in tokenized["tokenized_en"]
            ],
            "ids_fr": [
                self.word_to_id_fr.get(token, len(self.most_frequent_words_fr) - 1)
                for token in tokenized["tokenized_fr"]
            ],
        }


class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.languages = ["english", "french"]
        self.types = ["idx", "sentences"]

    def __len__(self):
        return len(
            self.data["english"]["idx"]
        )  # Assuming all language and data types have the same length

    def __getitem__(self, index):
        sample = {}
        for language in self.languages:
            sample[language] = {}
            for type in self.types:
                sample[language][type] = self.data[language][type][index]
        return sample


class to_tensor:
    """
    Transform class to convert word IDs to PyTorch tensors.
    """

    def __init__(self, tor):
        self.tor = tor

    def __call__(self, ids):
        """
        Convert word IDs to PyTorch tensors.
        """
        return {
            "ids_en": self.tor.tensor(ids["ids_en"]),
            "ids_fr": self.tor.tensor(ids["ids_fr"]),
        }


def load_data(train_len, val_len, kx=30000, ky=30000, Tx=30, Ty=30, batch_size=32):
    """
    Load and preprocess data for training and validation.
    """
    mt_en = MosesTokenizer(lang="en")
    mt_fr = MosesTokenizer(lang="fr")

    # Load English and French unigram frequency data
    bow_english = pd.read_csv("data/unigram_freq_en.csv")
    bow_french = pd.read_csv("data/unigram_freq_fr.csv")
    bow_english = bow_english[:kx]
    bow_french = bow_french[:ky]

    # Load WMT14 dataset
    wmt14 = load_dataset("wmt14", "fr-en", data_dir="data/")

    # Accessing example data
    train_data = wmt14["train"]
    val_data = wmt14["validation"]

    # Select a subset of data if specified
    if train_len is not None:
        train_data = train_data.select(range(train_len))
    else:
        train_len = len(train_data)
    if val_len is not None:
        val_data = val_data.select(range(val_len))
    else:
        val_len = len(val_data)

    tokenizer_wrapper = TokenizerWrapper(mt_en, mt_fr)

    # Tokenize and save train data if not already done
    if not os.path.exists("processed_data/tokenized_train_data{}".format(train_len)):
        tokenized_train_data = train_data.map(
            tokenizer_wrapper.tokenize_function,
            batched=False,
            num_proc=19,
            remove_columns=["translation"],
        )
        tokenized_train_data.save_to_disk(
            "processed_data/tokenized_train_data{}".format(train_len)
        )

    # Tokenize and save validation data if not already done
    if not os.path.exists("processed_data/tokenized_val_data{}".format(val_len)):
        tokenized_val_data = val_data.map(
            tokenizer_wrapper.tokenize_function,
            batched=False,
            num_proc=19,
            remove_columns=["translation"],
        )
        tokenized_val_data.save_to_disk(
            "processed_data/tokenized_val_data{}".format(val_len)
        )

    tokenized_train_data = load_from_disk(
        "processed_data/tokenized_train_data{}".format(train_len)
    )
    tokenized_val_data = load_from_disk(
        "processed_data/tokenized_val_data{}".format(val_len)
    )

    # Get most frequent English and French words
    most_frequent_english_words = bow_english["word"].apply(lambda x: str(x)).tolist()
    most_frequent_french_words = bow_french["word"].apply(lambda x: str(x)).tolist()
    tokenized_most_frequent_english_words = mt_en.tokenize(
        " ".join(most_frequent_english_words)
    )[: kx - 1]
    tokenized_most_frequent_french_words = mt_fr.tokenize(
        " ".join(most_frequent_french_words)
    )[: ky - 1]
    tokenized_most_frequent_english_words.append("<unk>")
    tokenized_most_frequent_french_words.append("<unk>")

    to_id_transform = toIdTransform(
        tokenized_most_frequent_english_words,
        tokenized_most_frequent_french_words,
        torch.tensor,
    )

    # Convert tokenized sentences to word IDs and save train data if not already done
    if not os.path.exists("processed_data/id_train_data{}".format(train_len)):
        tokenized_train_data = tokenized_train_data.map(
            to_id_transform, batched=False, num_proc=19
        )
        tokenized_train_data.save_to_disk(
            "processed_data/id_train_data{}".format(train_len)
        )

    # Convert tokenized sentences to word IDs and save validation data if not already done
    if not os.path.exists("processed_data/id_val_data{}".format(val_len)):
        tokenized_val_data = tokenized_val_data.map(
            to_id_transform, batched=False, num_proc=19
        )
        tokenized_val_data.save_to_disk("processed_data/id_val_data{}".format(val_len))

    tokenized_train_data = load_from_disk(
        "processed_data/id_train_data{}".format(train_len)
    )
    tokenized_val_data = load_from_disk("processed_data/id_val_data{}".format(val_len))

    # Helper function to pad sequences to a specified length
    def pad_to_length(x, length, pad_value):
        if len(x) < length:
            return x + [pad_value] * (length - len(x))
        else:
            return x[:length]

    # Initialize tensors for train and validation data
    idx_train_tensor_en = torch.zeros(
        (len(tokenized_train_data), Tx), dtype=torch.int16
    )
    idx_train_tensor_fr = torch.zeros(
        (len(tokenized_train_data), Ty), dtype=torch.int16
    )
    idx_val_tensor_en = torch.zeros((len(tokenized_val_data), Tx), dtype=torch.int16)
    idx_val_tensor_fr = torch.zeros((len(tokenized_val_data), Ty), dtype=torch.int16)

    # Pad sequences and convert to PyTorch tensors for train data
    for i, x in enumerate(tokenized_train_data):
        idx_train_tensor_en[i] = torch.tensor(pad_to_length(x["ids_en"], Tx, kx))
        idx_train_tensor_fr[i] = torch.tensor(pad_to_length(x["ids_fr"], Ty, ky))

    # Pad sequences and convert to PyTorch tensors for validation data
    for i, x in enumerate(tokenized_val_data):
        idx_val_tensor_en[i] = torch.tensor(pad_to_length(x["ids_en"], Tx, kx))
        idx_val_tensor_fr[i] = torch.tensor(pad_to_length(x["ids_fr"], Ty, ky))

    # Extract English and French sentences for train and validation data
    train_english_sentences = [
        train_data[i]["translation"]["en"] for i in range(len(train_data))
    ]
    train_french_sentences = [
        train_data[i]["translation"]["fr"] for i in range(len(train_data))
    ]
    val_english_sentences = [
        val_data[i]["translation"]["en"] for i in range(len(val_data))
    ]
    val_french_sentences = [
        val_data[i]["translation"]["fr"] for i in range(len(val_data))
    ]

    # Organize data into a dictionary
    data = dict(
        train=dict(
            english=dict(idx=idx_train_tensor_en, sentences=train_english_sentences),
            french=dict(idx=idx_train_tensor_fr, sentences=train_french_sentences),
        ),
        val=dict(
            english=dict(idx=idx_val_tensor_en, sentences=val_english_sentences),
            french=dict(idx=idx_val_tensor_fr, sentences=val_french_sentences),
        ),
        bow=dict(
            english=np.array(tokenized_most_frequent_english_words),
            french=np.array(tokenized_most_frequent_french_words),
        ),
    )
    train_dataset = TranslationDataset(data["train"])
    val_dataset = TranslationDataset(data["val"])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    return (
        (data["train"], train_dataloader),
        (data["val"], val_dataloader),
        (data["bow"]["english"], data["bow"]["french"]),
    )
