import os
import re
import glob

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from torchnlp.datasets import imdb_dataset
from torchnlp.download import download_file_maybe_extract
from torchnlp.encoders.text import WhitespaceEncoder

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize


def download_imdb_dataset(directory='data/',
                          train=False,
                          test=False,
                          train_directory='train',
                          test_directory='test',
                          extracted_name='aclImdb',
                          check_files=['aclImdb/README'],
                          url='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                          sentiments=['pos', 'neg']):
    """
    Load the IMDB dataset (Large Movie Review Dataset v1.0).

    This is a dataset for binary sentiment classification containing substantially more data than
    previous benchmark datasets. Provided a set of 25,000 highly polar movie reviews for
    training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text
    and already processed bag of words formats are provided.

    Note:
        The order examples are returned is not guaranteed due to ``iglob``.

    **Reference:** http://ai.stanford.edu/~amaas/data/sentiment/

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        train_directory (str, optional): The directory of the training split.
        test_directory (str, optional): The directory of the test split.
        extracted_name (str, optional): Name of the extracted dataset directory.
        check_files (str, optional): Check if these files exist, then this download was successful.
        url (str, optional): URL of the dataset ``tar.gz`` file.
        sentiments (list of str, optional): Sentiments to load from the dataset.

    Returns:
        :class:`tuple` of :class:`iterable` or :class:`iterable`:
        Returns between one and all dataset splits (train, dev and test) depending on if their
        respective boolean argument is ``True``.

    Example:
        >>> from torchnlp.datasets import imdb_dataset  # doctest: +SKIP
        >>> train = imdb_dataset(train=True)  # doctest: +SKIP
        >>> train[0:2]  # doctest: +SKIP
        [{
          'text': 'For a movie that gets no respect there sure are a lot of memorable quotes...',
          'sentiment': 'pos'
        }, {
          'text': 'Bizarre horror movie filled with famous faces but stolen by Cristina Raines...',
          'sentiment': 'pos'
        }]
    """
    download_file_maybe_extract(url=url, directory=directory, check_files=check_files)

    ret = []
    splits = [
        dir_ for (requested, dir_) in [(train, train_directory), (test, test_directory)]
        if requested
    ]
    for split_directory in splits:
        full_path = os.path.join(directory, extracted_name, split_directory)
        examples = []
        for sentiment in sentiments:
            for filename in glob.iglob(os.path.join(full_path, sentiment, '*.txt')):
                with open(filename, 'r', encoding="utf-8") as f:
                    text = f.readline()
                examples.append({
                    'text': text,
                    'sentiment': sentiment,
                })
        ret.append(examples)

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def import_dataset(cache_dir='./data', seq_length=2500):
    train_ds, test_ds = imdb_dataset(directory=cache_dir, train=True, test=True)

    # train_encoder = WhitespaceEncoder(train_ds)
    # test_encoder = WhitespaceEncoder(test_ds)

    print(f"dataset loaded")

    print(f"begin train_set tokenization")
    # train_set = [(word_tokenize(d['text']), d['sentiment']) for d in train_ds]
    # i = 0
    max_train_seq_len = 0
    # max_train_seq = ""
    train_set = []
    for d in train_ds:
        tokens = word_tokenize(remove_html(d['text']))
        if len(tokens) > max_train_seq_len:
            max_train_seq_len = len(tokens)
            # max_train_seq = tokens
        train_set.append((tokens, d['sentiment']))
        # i += 1
        # if i % 1000 == 0:
        #     print(f"train_set tokenize step:  {i}  ||  sentiment:  {d['sentiment']}")
        # print(f"sentiment:  {d['sentiment']}  ||  text:  {d['text']}")
    print(f"complete train_set tokenization")

    print(f"begin test_set tokenization")
    # test_set = [(word_tokenize(d['text']), d['sentiment']) for d in test_ds]
    i = 0
    max_test_seq_len = 0
    # max_test_seq = ""
    test_set = []
    for d in test_ds:
        tokens = word_tokenize(remove_html(d['text']))
        if len(tokens) > max_test_seq_len:
            max_test_seq_len = len(tokens)
            # max_test_seq = tokens
        test_set.append((tokens, d['sentiment']))
        # i += 1
        # if i % 1000 == 0:
        #   print(f"test_set tokenize step:  {i}  ||  sentiment:  {d['sentiment']}")
        #   print(f"sentiment:  {d['sentiment']}  ||  text:  {d['text']}")
    print(f"complete test_set tokenization")

    # print(f"max train set:   {max_train_seq}")
    # print(f"max test  set:   {max_test_seq}")

    print(f"max tokens train set:   {max_train_seq_len}")
    print(f"max tokens test  set:   {max_test_seq_len}")

    print(f"begin build_lexicon")
    index_word_lex, word_index_lex = build_lexicon([train_set, test_set])
    print(f"complete build_lexicon")

    print(f"begin train_enc build")
    train_enc = [(encode_and_pad(seq, seq_length, word_index_lex), label_mux(label)) for seq, label in train_set]
    print(f"complete train_enc build")

    print(f"begin test_enc build")
    test_enc = [(encode_and_pad(seq, seq_length, word_index_lex), label_mux(label)) for seq, label in test_set]
    print(f"complete test_enc build")

    print(f"begin build_dataloaders")
    train_dl, test_dl = build_dataloaders(train_enc, test_enc, batch_size=seq_length)
    print(f"complete build_dataloaders")

    return train_dl, test_dl, index_word_lex, word_index_lex


def remove_links(tweet):
    link_pattern = "https?:\/\/t.co/[\w]+"
    mention_pattern = "@\w+"
    tweet = re.sub(link_pattern, "", tweet)
    tweet = re.sub(mention_pattern, "", tweet)
    return tweet.lower()


def remove_html(seq):
    html_pattern = r'<.*?>'
    punc_set = r'[,.\/\\;:\"\"-_+=\[\]\{\}]'
    seq = re.sub(html_pattern, " ", seq)
    seq = re.sub(punc_set, " ", seq)
    return seq.lower()


def label_mux(label):
    if label == "neg":
        return 0
    elif label == "pos":
        return 2
    else:
        return 1


def build_lexicon(datasets):
    index_to_word_lex = {
        "<SOS>",
        "<EOS>",
        "<PAD>"
    }
    word_to_index_lex = {}

    i = 0
    for dataset in datasets:
        for seq, label in dataset:
            for token in seq:
                if token not in index_to_word_lex:
                    # index_to_word_lex.append(token)
                    # word_to_index_lex[token] = index_to_word_lex.index(token)
                    index_to_word_lex.add(token)
                    i += 1
                    if i % 10000 == 0:
                        print(f"build_lexicon step:  {i}  ||  token:  {token}")

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


def build_dataloaders(train_set_enc, test_set_enc, batch_size=500, shuffle=True, drop_last=True):
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
    download_imdb_dataset()
