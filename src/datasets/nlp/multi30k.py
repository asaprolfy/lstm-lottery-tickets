import os
import numpy as np
import pandas as pd

from torchnlp.download import download_files_maybe_extract

import torch
from torch.utils.data import TensorDataset, DataLoader
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def download_multi30k_dataset(directory='data/multi30k/',
                              train=False,
                              dev=False,
                              test=False,
                              train_filename='train',
                              dev_filename='val',
                              test_filename='test',
                              check_files=None,
                              urls=None):
    """
    Load the WMT 2016 machine translation dataset.

    As a translation task, this task consists in translating English sentences that describe an
    image into German, given the English sentence itself. As training and development data, we
    provide 29,000 and 1,014 triples respectively, each containing an English source sentence, its
    German human translation. As test data, we provide a new set of 1,000 tuples containing an
    English description.

    Status:
        Host ``www.quest.dcs.shef.ac.uk`` forgot to update their SSL
        certificate; therefore, this dataset does not download securely.

    References:
        * http://www.statmt.org/wmt16/multimodal-task.html
        * http://shannon.cs.illinois.edu/DenotationGraph/

    **Citation**
    :param urls:
    :param check_files:
    :param directory:
    :param train:
    :param dev:
    :param test:
    :param train_filename:
    :param dev_filename:
    :param test_filename:
    ::

        @article{elliott-EtAl:2016:VL16,
            author    = {{Elliott}, D. and {Frank}, S. and {Sima'an}, K. and {Specia}, L.},
            title     = {Multi30K: Multilingual English-German Image Descriptions},
            booktitle = {Proceedings of the 5th Workshop on Vision and Language},
            year      = {2016},
            pages     = {70--74},
            year      = 2016
        }

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the dev split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        train_directory (str, optional): The directory of the training split.
        dev_directory (str, optional): The directory of the dev split.
        test_directory (str, optional): The directory of the test split.
        check_files (str, optional): Check if these files exist, then this download was successful.
        urls (str, optional): URLs to download.

    Returns:
        :class:`tuple` of :class:`iterable` or :class:`iterable`:
        Returns between one and all dataset splits (train, dev and test) depending on if their
        respective boolean argument is ``True``.

    Example:
        >> from torchnlp.datasets import multi30k_dataset  # doctest: +SKIP
        >> train = multi30k_dataset(train=True)  # doctest: +SKIP
        >> train[:2]  # doctest: +SKIP
        [{
          'en': 'Two young, White males are outside near many bushes.',
          'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'
        }, {
          'en': 'Several men in hard hatsare operating a giant pulley system.',
          'de': 'Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.'
        }]
    """
    if check_files is None:
        check_files = ['train.de', 'val.de']
    if urls is None:
        urls = [
            'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
            'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
            'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz'
        ]
    # download_files_maybe_extract(urls=urls, directory=directory, check_files=check_files)

    ret = []
    splits = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    splits = [f for (requested, f) in splits if requested]

    for filename in splits:
        examples = []

        en_path = os.path.join(directory, filename + '.en')
        de_path = os.path.join(directory, filename + '.de')
        en_file = [line.strip() for line in open(en_path, 'r', encoding='utf-8')]
        de_file = [line.strip() for line in open(de_path, 'r', encoding='utf-8')]
        assert len(en_file) == len(de_file)
        for i in range(len(en_file)):
            if en_file[i] != '' and de_file[i] != '':
                examples.append({'en': en_file[i], 'de': de_file[i]})

        ret.append(examples)

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


if __name__ == '__main__':
    download_multi30k_dataset()
