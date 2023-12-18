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


def main(device, config, n_iter=10, batch_size=32, hidden_dim=32):

    res = []

    train_dl, test_dl, index_word_lex, word_index_lex = import_dataset(seq_length=batch_size)

    print(f"len train_dl:  {len(train_dl.dataset)}")
    # print(f"train_dl head:  {train_dl.dataset[0]}")

    model = BRNN_LSTM(batch_size, word_index_lex, num_layers=config['layers'], hidden_dim=hidden_dim)
    model.to(device)
    original_state_dict = copy.deepcopy(model.state_dict())
    print(f"original_state_dict")
    print(original_state_dict)

    torch.save(model,
               f"./saves/brnn_lstm/imdb/{config['lr']}_{config['layers']}_{config['epochs']}/init.pth.tar")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for i in range(0, n_iter):
        print(f"begin iteration:  {i}")

        sparsity_rates = []

        losses, accuracies = train(model, criterion, optimizer, train_dl, device, config['epochs'])

        test_result = evaluate(model, test_dl, device)

        if i + 1 < n_iter:
            torch.save(model,
                       f"./saves/brnn_lstm/imdb/{config['lr']}_{config['layers']}_{config['epochs']}/i_{i}.pth.tar")

            # mask, orig_wgt_tensors = create_mask(model, prune_rate=(float(i) / 10.0))
            mask, orig_wgt_tensors = create_apply_mask(model)

            reset_orig_wgt_tensors(model, mask, orig_wgt_tensors, original_state_dict)
            apply_mask(model, mask)

            print(f"mask:  ")
            print(mask)

            np_arr = np.array(mask).flatten()

            n_nonpruned_wgts = np.count_nonzero(np_arr)
            n_total_wgts = len(np_arr)
            n_pruned_wgts = n_total_wgts - n_nonpruned_wgts
            # sparsity_rate = (n_total_wgts - n_pruned_wgts) / n_total_wgts

            if n_total_wgts != 0:
                sparsity_rate = 1 - ((n_total_wgts - n_pruned_wgts) / n_total_wgts)
                sparsity_rates.append(sparsity_rate)
                print(f"sparsity_rate:  {sparsity_rate}")

            print(f"after prune iter:  {i}  ||  num_pruned:  {n_pruned_wgts}  ||  num_total_wgts:  {n_total_wgts}")

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


if __name__ == '__main__':
    filename = "./out/iter-results-0.json"

    dev = torch.device("mps")
    num_iterations = 10
    conf = {
        "lr": 1e-4,
        "layers": 4,
        "epochs": 15
    }

    results = main(dev, conf, num_iterations)

    with open(filename, 'w') as file:
        json.dump(file, results)
