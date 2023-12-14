import time

import torch
import torch.nn as nn

from datasets.sentiment.financial_data import import_dataset
from models.sentiment_lstm import Sentiment_LSTM
from train.sentiment.train import train
from evaluate.sentiment.test import evaluate


def main(data_filename):

    batch_size = 50
    device = torch.device("mps")
    learning_rate = 3e-4

    train_dl, test_dl, index_word_lex, word_index_lex = import_dataset(data_filename)

    model = Sentiment_LSTM(batch_size, word_index_lex)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(model, criterion, optimizer, train_dl, device)

    evaluate(model, test_dl, device)


if __name__ == '__main__':
    filename = "../resources/datasets/sentiment/financialdata.csv"
    main(filename)
