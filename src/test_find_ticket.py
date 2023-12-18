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


def main(device, model, train_dl, test_dl, index_word_lex, word_index_lex, config, iter, batch_size=32, hidden_dim=32):

    res = []

    model_0 = copy.deepcopy(model)

    model_1 = copy.deepcopy(model_0)

    torch.save(model_0,
               f"./saves/brnn_lstm/imdb/{config['lr']}_{config['layers']}_{config['epochs']}/init.pth.tar")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_0.parameters(), lr=config['lr'])

    losses, accuracies = train(model_0, criterion, optimizer, train_dl, device, config['epochs'])

    print(f"test result for original model: ")
    test_result = evaluate(model_0, test_dl, device)

    # rate = 0.1 * iter

    # match iter:
    #     case 9:
    #         rate = 0.85
    #     case 10:
    #         rate = 0.90
    #     case 11:
    #         rate = 0.95
    #     case _:
    #         rate = 0.1 * iter

    match iter:
        case 10:
            rate = 0.92
        case 11:
            rate = 0.94
        case 12:
            rate = 0.96
        case 13:
            rate = 0.98
        case _:
            rate = iter * 0.1

    mask, orig_wgt_tensors, unpruned = create_mask_dict(model_0, prune_rate=rate)
    apply_mask_dict(model_1, mask)

    criterion_p = nn.CrossEntropyLoss()
    optimizer_p = torch.optim.Adam(model_1.parameters(), lr=config['lr'])

    losses, accuracies = train(model_1, criterion_p, optimizer_p, train_dl, device, config['epochs'])

    print(f"test result for ticket model: ")
    test_result_p = evaluate(model_1, test_dl, device)

    total = 0
    pruned = 0

    for layer in mask:
        for param in mask[layer]:
            print(layer)
            print(param)
            # print(mask[str(layer)][str(param)])
            # total += count_items(mask[k])
            # unpruned += np.count_nonzero(mask[k])
            c, z = count_items(mask[str(layer)][str(param)])
            total += c
            pruned += z

    sparsity = pruned / total

    print(f"after prune iter:  {iter}  ||  num_pruned:  {pruned}  ||  num_total_wgts:  {total}")
    print(f"sparsity:  {sparsity}")

    model_random = BRNN_LSTM(batch, word_index, num_layers=conf['layers'], hidden_dim=hidden)
    model_random.to(dev)
    apply_mask_dict(model_random, mask)
    criterion_random = nn.CrossEntropyLoss()
    optimizer_random = torch.optim.Adam(model_random.parameters(), lr=config['lr'])

    losses_random, accuracies_random = train(model_random, criterion_random, optimizer_random,
                                             train_dl, device, config['epochs'])

    print(f"test result for ticket model: ")
    test_result_random = evaluate(model_random, test_dl, device)

    res = {
        "CONFIG": config,
        "ITERATIONS": f"{iter}",
        "RESULTS": f"{test_result_p}",
        "LOSSES": f"{losses}",
        "ACCURACIES": f"{accuracies}",
        "SPARSITY": f"{sparsity}",
        "TEST_RESULT_ORIGINAL": f"{test_result}",
        "TEST_RESULT_TICKET": f"{test_result_p}",
        "TEST_RESULT_RANDOM_PRUNED": f"{test_result_random}"
    }

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
    batch = 32
    hidden = 32

    # device, model, train_dl, test_dl, index_word_lex, word_index_lex

    train_d, test_d, index_word, word_index = import_dataset(seq_length=batch)
    print(f"len train_dl:  {len(train_d.dataset)}")
    model_a = BRNN_LSTM(batch, word_index, num_layers=conf['layers'], hidden_dim=hidden)
    model_a.to(dev)

    results = []

    # for i in range(0, 12):
    for i in range(5, 14):
        ress = main(dev, model_a, train_d, test_d, index_word, word_index, conf, i, batch_size=batch)
        results.append(ress)

    with open(filename, 'w') as file:
        # file.write(str(results))
        json.dump(results, file)
