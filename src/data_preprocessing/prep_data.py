import os
from multiprocessing import Manager, Process, cpu_count
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from sacremoses import MosesTokenizer
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from global_variables import DATA_DIR, EXT_DATA_DIR

n_processors = cpu_count()


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

        tokenized_en = self.tokenizer_en.tokenize(preprocessed_en)
        tokenized_fr = self.tokenizer_fr.tokenize(preprocessed_fr)

        # length_en = len(tokenized_en)
        # length_fr = len(tokenized_fr)

        return {
            "tokenized_en": tokenized_en,
            "tokenized_fr": tokenized_fr,
            # "Tx": length_en,
            # "Ty": length_fr,
        }


class toWordCount:
    """
    Transform class to convert tokenized sentences to corresponding word IDs.
    """

    def __init__(self, cont):
        self.cont = cont

    def __call__(self, tokenized):
        count_en = self.cont()
        count_fr = self.cont()
        count_en.update(tokenized["tokenized_en"])
        count_fr.update(tokenized["tokenized_fr"])

        return {
            "count_en_words": list(count_en.keys()),
            "count_en_freq": list(count_en.values()),
            "count_fr_words": list(count_fr.keys()),
            "count_fr_freq": list(count_fr.values()),
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


from collections import Counter


def extract_word_frequency_dicho(data):
    if len(data) == 1:
        dictionary = {
            "en": Counter(data[0]["count_en_words"]),
            "fr": Counter(data[0]["count_fr_words"]),
        }
        df_en = pd.DataFrame(dictionary["en"].items(), columns=["word", "freq"])
        df_fr = pd.DataFrame(dictionary["fr"].items(), columns=["word", "freq"])
        return df_en, df_fr
    if len(data) == 0:
        df_en = pd.DataFrame(columns=["word", "freq"])
        df_fr = pd.DataFrame(columns=["word", "freq"])
        return df_en, df_fr
    middle = len(data) // 2

    left_df_en, left_df_fr = extract_word_frequency_dicho(data.select(range(middle)))
    right_df_en, right_df_fr = extract_word_frequency_dicho(
        data.select(range(middle, len(data)))
    )

    # combine the two dataframes and sum the frequencies
    df_en = pd.concat([left_df_en, right_df_en])
    df_en = df_en.groupby("word").sum().reset_index()
    df_fr = pd.concat([left_df_fr, right_df_fr])
    df_fr = df_fr.groupby("word").sum().reset_index()

    df_en.sort_values(by=["freq"], ascending=False, inplace=True)
    df_fr.sort_values(by=["freq"], ascending=False, inplace=True)

    return df_en, df_fr


import string


def extract_word_frequency(data):
    word_freq = {"en": Counter(), "fr": Counter()}

    for i, x in enumerate(data):
        for lang in ["en", "fr"]:
            word_freq_dict = dict(
                zip(
                    x["count_{}_words".format(lang)],
                    x["count_{}_freq".format(lang)],
                )
            )
            # Exclude simple punctuation marks
            word_freq_dict = {
                word: freq
                for word, freq in word_freq_dict.items()
                if word not in string.punctuation
            }
            word_freq[lang].update(word_freq_dict)
        if (i / len(data)) % 0.1 == 0:
            print("Processed {} / {} samples".format(i + 1, len(data)), end="\r")
    df_en = pd.DataFrame(word_freq["en"].items(), columns=["word", "freq"])
    df_fr = pd.DataFrame(word_freq["fr"].items(), columns=["word", "freq"])

    df_en.sort_values(by=["freq"], ascending=False, inplace=True)
    df_fr.sort_values(by=["freq"], ascending=False, inplace=True)

    return df_en, df_fr


# Helper function to pad sequences to a specified length
def pad_to_length(x, length, pad_value):
    if len(x) < length:
        return x + [pad_value] * (length - len(x))
    else:
        return x[:length]


# Function to process train data in parallel
def process_data(tokenized, Tx, kx, Ty, ky, idx_tensor_en, idx_tensor_fr):
    for i, x in enumerate(tokenized):
        idx_tensor_en[i] = torch.tensor(pad_to_length(x["ids_en"], Tx, kx))
        idx_tensor_fr[i] = torch.tensor(pad_to_length(x["ids_fr"], Ty, ky))


def pad_multiprocess(data, idx_en, idx_fr, Tx, Ty, kx, ky, multiprocess=True):
    if multiprocess == True:
        jobs = []
        for i in range(n_processors):
            start, end = (
                i * len(data) // n_processors,
                (i + 1) * len(data) // n_processors,
            )
            p = Process(
                target=process_data,
                args=(
                    data.select(list(range(start, end))),
                    Tx,
                    kx,
                    Ty,
                    ky,
                    idx_en[start:end],
                    idx_fr[start:end],
                ),
            )
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
    else:
        process_data(data, Tx, kx, Ty, ky, idx_en, idx_fr)


# import autotokenizer
def load_data(
    train_len,
    val_len,
    kx=30000,
    ky=30000,
    Tx=30,
    Ty=30,
    batch_size=32,
    tokenizer="Moses",
    vocab_source="train",
    mp=True,
    only_vocab=False,
):
    """
    Load and preprocess data for training and validation.
    """

    print("Loading and preprocessing data...")
    mt_en = (
        MosesTokenizer(lang="en")
        if tokenizer == "Moses"
        else AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    )
    mt_fr = (
        MosesTokenizer(lang="fr")
        if tokenizer == "Moses"
        else AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    )

    if not only_vocab:
        # Load WMT14 dataset
        wmt14 = load_dataset("wmt14", "fr-en", data_dir="data/")

        # Accessing example data
        train_data = wmt14["train"]
        val_data = wmt14["validation"]

        # shuffle data
        train_data = train_data.shuffle(seed=42)
        val_data = val_data.shuffle(seed=42)

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
        if not os.path.exists(
            DATA_DIR / "processed_data/tokenized_train_data_{}".format(train_len)
        ):
            print("Tokenizing train data...")
            tokenized_train_data = train_data.map(
                tokenizer_wrapper.tokenize_function,
                batched=False,
                num_proc=n_processors,
                remove_columns=["translation"],
            )
            tokenized_train_data.save_to_disk(
                DATA_DIR / "processed_data/tokenized_train_data_{}".format(train_len)
            )

        # Tokenize and save validation data if not already done
        if not os.path.exists(
            DATA_DIR / "processed_data/tokenized_val_data_{}".format(val_len)
        ):
            print("Tokenizing validation data...")
            tokenized_val_data = val_data.map(
                tokenizer_wrapper.tokenize_function,
                batched=False,
                num_proc=n_processors,
                remove_columns=["translation"],
            )
            tokenized_val_data.save_to_disk(
                DATA_DIR / "processed_data/tokenized_val_data_{}".format(val_len)
            )

        tokenized_train_data = load_from_disk(
            DATA_DIR / "processed_data/tokenized_train_data_{}".format(train_len)
        )
        tokenized_val_data = load_from_disk(
            DATA_DIR / "processed_data/tokenized_val_data_{}".format(val_len)
        )

        if not os.path.exists(
            DATA_DIR / "processed_data/word_count_{}".format(train_len)
        ):
            print("Counting word frequency...")
            word_count_train = tokenized_train_data.map(
                toWordCount(Counter), batched=False, num_proc=n_processors
            )
            word_count_val = tokenized_val_data.map(
                toWordCount(Counter), batched=False, num_proc=n_processors
            )
            word_count = concatenate_datasets([word_count_train, word_count_val])
            word_count.save_to_disk(
                DATA_DIR / "processed_data/word_count_{}".format(train_len)
            )
        word_count = load_from_disk(
            DATA_DIR / "processed_data/word_count_{}".format(train_len)
        )

    print("Extracting word frequency...")
    if os.path.exists(DATA_DIR / "dictionaries/") == False:
        os.mkdir(DATA_DIR / "dictionaries/")
    if vocab_source == "train":
        if not os.path.exists(
            DATA_DIR / "dictionaries/unigram_freq_en_{}.csv".format(train_len)
        ):
            if only_vocab:
                raise ValueError(
                    "No vocabulary file found, please rerun with only_vocab = False to extract"
                )
            df_en, df_fr = extract_word_frequency(word_count)
            df_en.to_csv(
                DATA_DIR / "dictionaries/unigram_freq_en_{}.csv".format(train_len),
                index=False,
            )
            df_fr.to_csv(
                DATA_DIR / "dictionaries/unigram_freq_fr_{}.csv".format(train_len),
                index=False,
            )
        df_en = pd.read_csv(
            DATA_DIR / "dictionaries/unigram_freq_en_{}.csv".format(train_len)
        )
        df_fr = pd.read_csv(
            DATA_DIR / "dictionaries/unigram_freq_fr_{}.csv".format(train_len)
        )
        bow_english, bow_french = df_en, df_fr
    else:
        bow_english = pd.read_csv(EXT_DATA_DIR / "dictionaries/unigram_freq_en_ext.csv")
        bow_french = pd.read_csv(EXT_DATA_DIR / "dictionaries/unigram_freq_fr_ext.csv")
    print("Done extraction")
    bow_english = bow_english[:kx]
    bow_french = bow_french[:ky]

    # Get most frequent English and French words
    most_frequent_english_words = bow_english["word"].apply(lambda x: str(x)).tolist()
    most_frequent_french_words = bow_french["word"].apply(lambda x: str(x)).tolist()
    tokenized_most_frequent_english_words = mt_en.tokenize(
        " ".join(most_frequent_english_words)
    )[: kx - 1]
    tokenized_most_frequent_french_words = mt_fr.tokenize(
        " ".join(most_frequent_french_words)
    )[: ky - 1]
    tokenized_most_frequent_english_words.append("")
    tokenized_most_frequent_french_words.append("")

    if not only_vocab:
        to_id_transform = toIdTransform(
            tokenized_most_frequent_english_words,
            tokenized_most_frequent_french_words,
            torch.tensor,
        )

        # Convert tokenized sentences to word IDs and save train data if not already done
        if not os.path.exists(
            DATA_DIR
            / "processed_data/id_train_data_{}_{}_{}_{}".format(
                train_len, kx, ky, vocab_source
            )
        ):
            print("Converting train data to word IDs...")
            tokenized_train_data = tokenized_train_data.map(
                to_id_transform, batched=False, num_proc=n_processors
            )
            tokenized_train_data.save_to_disk(
                DATA_DIR
                / "processed_data/id_train_data_{}_{}_{}_{}".format(
                    train_len, kx, ky, vocab_source
                )
            )

        # Convert tokenized sentences to word IDs and save validation data if not already done
        if not os.path.exists(
            DATA_DIR
            / "processed_data/id_val_data_{}_{}_{}_{}".format(
                val_len, kx, ky, vocab_source
            )
        ):
            print("Converting validation data to word IDs...")
            tokenized_val_data = tokenized_val_data.map(
                to_id_transform, batched=False, num_proc=n_processors
            )
            tokenized_val_data.save_to_disk(
                DATA_DIR
                / "processed_data/id_val_data_{}_{}_{}_{}".format(
                    val_len, kx, ky, vocab_source
                )
            )

        tokenized_train_data = load_from_disk(
            DATA_DIR
            / "processed_data/id_train_data_{}_{}_{}_{}".format(
                train_len, kx, ky, vocab_source
            )
        )
        tokenized_val_data = load_from_disk(
            DATA_DIR
            / "processed_data/id_val_data_{}_{}_{}_{}".format(
                val_len, kx, ky, vocab_source
            )
        )

        # Initialize tensors for train and validation data
        idx_train_tensor_en = torch.zeros(
            (len(tokenized_train_data), Tx), dtype=torch.int16
        )
        idx_train_tensor_fr = torch.zeros(
            (len(tokenized_train_data), Ty), dtype=torch.int16
        )
        idx_val_tensor_en = torch.zeros(
            (len(tokenized_val_data), Tx), dtype=torch.int16
        )
        idx_val_tensor_fr = torch.zeros(
            (len(tokenized_val_data), Ty), dtype=torch.int16
        )

        # split the tensors according to the number of processors
        pad_multiprocess(
            tokenized_train_data,
            idx_train_tensor_en,
            idx_train_tensor_fr,
            Tx,
            Ty,
            kx,
            ky,
            mp,
        )
        pad_multiprocess(
            tokenized_val_data, idx_val_tensor_en, idx_val_tensor_fr, Tx, Ty, kx, ky, mp
        )

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
    else:
        idx_train_tensor_en = torch.zeros((1, Tx), dtype=torch.int16)
        idx_train_tensor_fr = torch.zeros((1, Ty), dtype=torch.int16)
        idx_val_tensor_en = torch.zeros((1, Tx), dtype=torch.int16)
        idx_val_tensor_fr = torch.zeros((1, Ty), dtype=torch.int16)
        train_english_sentences = []
        train_french_sentences = []
        val_english_sentences = []
        val_french_sentences = []
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

    # print some samples
    print("Some samples from the training dataset from the loaded data")
    for i, sample in enumerate(train_dataset):
        if i == 5:
            break
        print("English: ", sample["english"]["sentences"])
        print("French: ", sample["french"]["sentences"])
        print("English ids: ", sample["english"]["idx"])
        print("French ids: ", sample["french"]["idx"])
        print("\n")

    print("Some samples from the validation dataset from the loaded data")
    for i, sample in enumerate(val_dataset):
        if i == 5:
            break
        print("English: ", sample["english"]["sentences"])
        print("French: ", sample["french"]["sentences"])
        print("English ids: ", sample["english"]["idx"])
        print("French ids: ", sample["french"]["idx"])
        print("\n")

    print("Data loading and preprocessing complete.")

    return (
        (data["train"], train_dataloader),
        (data["val"], val_dataloader),
        (data["bow"]["english"], data["bow"]["french"]),
    )
