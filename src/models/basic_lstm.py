import numpy as np

import torch
import torch.nn as nn


class Basic_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_features, dropout_p):
        super(Basic_LSTM, self).__init__()
        self.hidden = None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.dropout_p = dropout_p

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim, out_features)
        self.softmax = nn.Softmax()

        self.layer_names = [

        ]

    def forward(self, x, hidden):
        # self.hidden = self.init_hidden(x.size(-1))
        
        embeddings = self.embedding(x)

        x, hidden = self.lstm(embeddings, hidden)
        out = self.dropout()
        out = self.fc(out)
        out = self.softmax(out)

        # x, (h, c) = self.lstm(embeddings, hidden)
        # out = self.dropout(h[-1])
        # out = self.fc(out)
        # out = self.softmax(out)

        return out, hidden

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.zeros(1, batch_size, 32), torch.zeros(1, batch_size, 32)


if __name__ == '__main__':
    print('Not a main file')
