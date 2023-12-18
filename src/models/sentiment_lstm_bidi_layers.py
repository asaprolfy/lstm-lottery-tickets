import torch
import torch.nn as nn
import torch.nn.init as init
import torchtext.vocab as vocab
import numpy as np

class BRNN_LSTM(nn.Module):
    def __init__(self, batch_size, word_to_index_lex,
                 hidden_dim=32, dropout=0.1, embedding_dim=100, num_layers=2):
        super(BRNN_LSTM, self).__init__()
        self.batch_size = batch_size
        self.word_to_index_lex = word_to_index_lex
        self.vocab_size = len(word_to_index_lex)
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        self.embedding_dim = embedding_dim
        self.num_directions = 2
        self.num_layers = num_layers

        # Initialize the embedding layer with pre-trained word vectors
        emb = self.init_embeddings()
        self.embedding = nn.Embedding.from_pretrained(emb)

        # The RNN layer takes in the embedding size and the hidden vector size.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True,)

        # We use dropout before the final layer to improve with regularization.
        self.dropout = nn.Dropout(dropout)

        # The fully-connected layer takes in the hidden dim of the RNN and
        # outputs a 3x1 vector of the class scores.
        self.fc = nn.Linear(hidden_dim * self.num_directions, 3)

        # self.init_weights()

    def forward(self, x, hidden):
        """
        The forward method takes in the input and the previous hidden state
        """

        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.fc(out)

        # We extract the scores for the final hidden state since it is the one that matters.
        out = out[:, -1, :]
        return out, hidden

    def init_hidden(self):
        return (torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim))

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

    def init_weights(self):
        for module in self.modules():
            match type(module):
                case nn.LSTM:
                    for p in module.parameters():
                        if len(p.shape) >= 2:
                            init.orthogonal_(p.data)
                        else:
                            init.normal_(p.data)
                case nn.LSTMCell:
                    for p in module.parameters():
                        if len(p.shape) >= 2:
                            init.orthogonal_(p.data)
                        else:
                            init.normal_(p.data)
                case nn.Linear:
                    init.xavier_normal_(module.weight.data)
                    init.normal_(module.bias.data)


if __name__ == '__main__':
    print('Not a main file')
