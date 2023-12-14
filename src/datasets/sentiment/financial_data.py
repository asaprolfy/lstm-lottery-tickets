import string
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import TensorDataset, DataLoader
from nltk.tokenize import word_tokenize


def import_dataset(filename='financialdata.csv', dataset_split=0.2, random_seed=19, seq_length=40):
    base_df = pd.read_csv(filename)

    train_df, test_df = train_test_split(base_df, test_size=dataset_split, random_state=random_seed)

    train_list = list(train_df.to_records(index=False))
    test_list = list(test_df.to_records(index=False))

    train_set = [(word_tokenize(remove_links(tweet)), label) for tweet, label in train_list]
    test_set = [(word_tokenize(remove_links(tweet)), label) for tweet, label in test_list]

    index_word_lex, word_index_lex = build_lexicon([train_set, test_set])

    train_enc = [(encode_and_pad(seq, seq_length, word_index_lex), label_mux(label)) for seq, label in train_set]
    test_enc = [(encode_and_pad(seq, seq_length, word_index_lex), label_mux(label)) for seq, label in test_set]

    train_dl, test_dl = build_dataloaders(train_enc, test_enc)

    return train_dl, test_dl, index_word_lex, word_index_lex


def remove_links(tweet):
    link_pattern = "https?:\/\/t.co/[\w]+"
    mention_pattern = "@\w+"
    tweet = re.sub(link_pattern, "", tweet)
    tweet = re.sub(mention_pattern, "", tweet)
    return tweet.lower()


def label_mux(label):
    if label == "negative":
        return 0
    elif label == "neutral":
        return 1
    else:
        return 2


def build_lexicon(datasets):
    index_to_word_lex = ["<SOS>", "<EOS>", "<PAD>"]
    word_to_index_lex = {}

    for dataset in datasets:
        for seq, label in dataset:
            for token in seq:
                if token not in index_to_word_lex:
                    index_to_word_lex.append(token)
                    # word_to_index_lex[token] = index_to_word_lex.index(token)

    word_to_index_lex = {token: i for i, token in enumerate(index_to_word_lex)}

    return index_to_word_lex, word_to_index_lex


def encode_and_pad(seq, length, word_to_index_lex):
    sos = [word_to_index_lex["<SOS>"]]
    eos = [word_to_index_lex["<EOS>"]]
    pad = [word_to_index_lex["<PAD>"]]

    if len(seq) < length - 2:  # -2 for SOS and EOS
        n_pads = length - 2 - len(seq)
        encoded = [word_to_index_lex[w] for w in seq]
        return sos + encoded + eos + pad * n_pads
    else:
        encoded = [word_to_index_lex[w] for w in seq]
        truncated = encoded[:length - 2]
        return sos + truncated + eos


def build_dataloaders(train_set_enc, test_set_enc, batch_size=50, shuffle=True, drop_last=True):
    train_x = np.array([seq for seq, label in train_set_enc])
    train_y = np.array([label for seq, label in train_set_enc])
    test_x = np.array([seq for seq, label in test_set_enc])
    test_y = np.array([label for seq, label in test_set_enc])

    train_tds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_tds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_dl = DataLoader(train_tds, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)
    test_dl = DataLoader(test_tds, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)

    return train_dl, test_dl


if __name__ == '__main__':
    print('Not a main file')
