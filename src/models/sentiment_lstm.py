import numpy as np

import torch
import torch.nn as nn
import torchtext.vocab as vocab


class Sentiment_LSTM(nn.Module):
    def __init__(self, batch_size, word_to_index_lex,
                 hidden_dim=32, dropout=0.1, embedding_dim=100, num_layers=1):
        super(Sentiment_LSTM, self).__init__()
        self.batch_size = batch_size
        self.word_to_index_lex = word_to_index_lex
        self.vocab_size = len(word_to_index_lex)
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        self.embedding_dim = embedding_dim
        emb = self.init_embeddings()
        self.embedding = nn.Embedding.from_pretrained(emb)
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x, hidden):
        """
        The forward method takes in the input and the previous hidden state
        """

        embeddings = self.embedding(x)

        # The embedded inputs are fed to the LSTM alongside the previous hidden state
        out, hidden = self.lstm(embeddings, hidden)

        # Dropout is applied to the output and fed to the FC layer
        out = self.dropout(out)
        out = self.fc(out)

        # We extract the scores for the final hidden state since it is the one that matters.
        out = out[:, -1, :]
        return out, hidden

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def count_parameters(self):
        return sum(param.numel() for param in self.parameters() if self.requires_grad_())

    def init_embeddings(self):
        glove = vocab.GloVe(name='twitter.27B', dim=self.embedding_dim)
        embeddings = torch.FloatTensor(self.vocab_size, self.embedding_dim)
        for word, i in self.word_to_index_lex.items():
            if word in glove.stoi:
                embeddings[i] = glove.vectors[glove.stoi[word]]
            else:
                embeddings[i] = torch.zeros(self.embedding_dim)
        return embeddings


if __name__ == '__main__':
    print('Not a main file')
