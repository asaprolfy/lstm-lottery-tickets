# import time
import copy
import json

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.utils.prune as prune

from datasets.nlp.imdb import import_dataset

from train.train import train
from evaluate.test import evaluate
from models.sentiment_lstm_bidi_layers import BRNN_LSTM
from pruning.misc_pruning_methods import apply_threshold_prune_all_layers as prune_all
from pruning.misc_pruning_methods import apply_l1un_prune_all_layers as prune_all_l1un

from pruning.layerwise_weight_prune import create_mask, apply_mask, create_apply_mask
from pruning.reset_prune import reset_model, reset_non_mask, reset_orig_wgt_tensors


def main(device, layers, n_iter=10, batch_size=32, hidden_dim=32):
    train_dl, test_dl, index_word_lex, word_index_lex = import_dataset(seq_length=batch_size)

    model = BRNN_LSTM(batch_size, word_index_lex, num_layers=layers, hidden_dim=hidden_dim)
    model.to(device)
    original_state_dict = copy.deepcopy(model.state_dict())

    mask, orig_wgt_tensors = create_apply_mask(model)
    reset_orig_wgt_tensors(model, mask, orig_wgt_tensors, original_state_dict)
    apply_mask(model, mask)


if __name__ == '__main__':
    num_layers = 2
    dev = torch.device("mps")
    main(dev, num_layers)
