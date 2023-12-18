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

from pruning.layerwise_weight_prune import create_mask_simple, apply_mask_simple
from pruning.layerwise_weight_prune import apply_mask_dict, create_mask_dict
from pruning.reset_prune import reset_model, reset_non_mask, reset_orig_wgt_tensors, reset_orig_wgt_tensors_dict


def main(device, layers, n_iter=10, batch_size=32, hidden_dim=32):
    train_dl, test_dl, index_word_lex, word_index_lex = import_dataset(seq_length=batch_size)

    model = BRNN_LSTM(batch_size, word_index_lex, num_layers=layers, hidden_dim=hidden_dim)
    model.to(device)
    original_state_dict = copy.deepcopy(model.state_dict())

    # mask, orig_wgt_tensors = create_mask_simple(model)
    # reset_orig_wgt_tensors_simple(model, mask, orig_wgt_tensors, original_state_dict)
    # apply_mask_simple(model, mask)

    mask, orig_wgt_tensors, unpruned = create_mask_dict(model)
    reset_orig_wgt_tensors_dict(model, mask, orig_wgt_tensors, original_state_dict)
    apply_mask_dict(model, mask)

    total = 0
    pruned = 0
    # unpruned = 0

    for layer in mask:
        for param in mask[layer]:
            print(layer)
            print(param)
            print(mask[str(layer)][str(param)])
            # total += count_items(mask[k])
            # unpruned += np.count_nonzero(mask[k])
            c, z = count_items(mask[str(layer)][str(param)])
            total += c
            pruned += z

    sparsity = pruned / total

    print(f"after prune iter:  {1}  ||  pruned:  {pruned} ||  unpruned:  {unpruned}  ||  num_total_wgts:  {total}")
    print(f"unpruned + pruned:  {pruned + unpruned}")
    print(f"sparsity:  {sparsity}")
    print(f"test pruned + unpruned == total :  {(pruned + unpruned) == total}")


def count_items(array):
    """Counts the number of items in an array.

    Args:
      array: The array to count the items in.

    Returns:
      The number of items in the array.
    """
    count = 0
    zeros = 0

    for v in array:
        if type(v) is array:
            c, z = count_items(v)
            count += c
            zeros += z
        elif type(v) is np.ndarray:
            c, z = count_items(v)
            count += c
            zeros += z
        else:
            count += 1
            if v == 0:
                zeros += 1
    return count, zeros


if __name__ == '__main__':
    num_layers = 2
    dev = torch.device("mps")
    main(dev, num_layers)
