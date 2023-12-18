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

from pruning.layerwise_weight_prune import create_mask_dict, apply_mask_dict, create_apply_mask_dict
from pruning.reset_prune import reset_orig_wgt_tensors_dict


def main(device, config, n_iter=10, batch_size=32, hidden_dim=32):

    res = []

    train_dl, test_dl, index_word_lex, word_index_lex = import_dataset(seq_length=batch_size)

    print(f"len train_dl:  {len(train_dl.dataset)}")
    # print(f"train_dl head:  {train_dl.dataset[0]}")

    model = BRNN_LSTM(batch_size, word_index_lex, num_layers=config['layers'], hidden_dim=hidden_dim)
    model.to(device)
    original_state_dict = copy.deepcopy(model.state_dict())
    print(f"original_state_dict nonzero num")
    # print(original_state_dict)
    original_nonzero_count = 0

    torch.save(model,
               f"./saves/brnn_lstm/imdb/{config['lr']}_{config['layers']}_{config['epochs']}/init.pth.tar")

    for i in range(0, n_iter):
        print(f"begin iteration:  {i}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        sparsity_rates = []

        losses, accuracies = train(model, criterion, optimizer, train_dl, device, config['epochs'])

        test_result = evaluate(model, test_dl, device)

        if i + 1 < n_iter:
            torch.save(model,
                       f"./saves/brnn_lstm/imdb/{config['lr']}_{config['layers']}_{config['epochs']}/i_{i}.pth.tar")

            # mask, orig_wgt_tensors = create_mask(model, prune_rate=(float(i) / 10.0))
            # mask, orig_wgt_tensors, unpruned = create_mask_dict(model)
            mask, orig_wgt_tensors, unpruned = create_apply_mask_dict(model)

            # reset_orig_wgt_tensors_dict(model, mask, orig_wgt_tensors, original_state_dict)
            # apply_mask_dict(model, mask)

            total = 0
            pruned = 0

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

            if original_nonzero_count == 0:
                original_nonzero_count = total

            sparsity = pruned / total

            sparsity_rates.append(sparsity)

            print(f"after prune iter:  {i}  ||  num_pruned:  {pruned}  ||  num_total_wgts:  {total}")
            print(f"sparsity:  {sparsity}")

        res.append(
            {
                "CONFIG": config,
                "ITERATIONS": n_iter,
                "RESULTS": test_result,
                "LOSSES": losses,
                "ACCURACIES": accuracies,
                "SPARSITIES": sparsity_rates
            }
        )

    return res


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
    filename = "./out/iter-results-0.json"

    dev = torch.device("mps")
    num_iterations = 10
    conf = {
        "lr": 1e-4,
        "layers": 2,
        "epochs": 10
    }

    results = main(dev, conf, num_iterations)

    with open(filename, 'w') as file:
        json.dump(file, results)
