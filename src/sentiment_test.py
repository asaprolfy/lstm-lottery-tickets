# import time
import copy
import json

import torch
import torch.nn as nn
# import torch.nn.utils.prune as prune

from datasets.sentiment.financial_data import import_dataset
from train.train import train
from evaluate.test import evaluate

# from models.sentiment_lstm import Sentiment_LSTM
from models.sentiment_lstm_bidi_layers import BRNN_LSTM
from pruning.misc_pruning_methods import apply_threshold_prune_all_layers as prune_all
from pruning.misc_pruning_methods import apply_l1un_prune_all_layers as prune_all_l1un


# pruning_rates = [
#     0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.45,
#     0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
#     0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99,
#     0.999
# ]

pruning_rates = [x / 100 for x in range(5, 99)]
pruning_rates.append(0.995)
pruning_rates.append(0.999)


def main(device, data_filename, num_layers=4, batch_size=50, learning_rate=3e-5, n_epochs=100):

    train_dl, test_dl, index_word_lex, word_index_lex = import_dataset(data_filename)

    accuracy_results = {}

    # model = Sentiment_LSTM(batch_size, word_index_lex)
    model = BRNN_LSTM(batch_size, word_index_lex, num_layers=num_layers)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(model, criterion, optimizer, train_dl, device, num_epochs=n_epochs)

    print(f"NUM LAYERS:  {num_layers}  |  NUM EPOCHS:  {n_epochs}  |  LEARN RATE:  {learning_rate}")

    print('Non pruned run:')
    evaluate(model, test_dl, device)

    for p_rate in pruning_rates:
        model_p = copy.deepcopy(model)
        prune_all_l1un(model_p, p_rate)
        print(f"{p_rate} prune test run: ")
        accuracy_results[p_rate] = evaluate(model_p, test_dl, device)

    del model, model_p
    return accuracy_results


if __name__ == '__main__':
    filename = "../resources/datasets/sentiment/financialdata.csv"

    out_filename = "./results-init-weights.json"

    layers = [2, 4, 6, 8]
    # learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2]
    learning_rates = [1e-4, 1e-3, 1e-2]
    n_epoch = [50]

    res = []

    dev = torch.device("mps")

    # main(dev, filename, num_layers=6)
    # main(dev, filename, num_layers=8)
    # main(dev, filename, num_layers=10)
    # main(dev, filename, num_layers=14, learning_rate=3e-5)
    # main(dev, filename, num_layers=14, learning_rate=4e-5)
    # main(dev, filename, num_layers=14, learning_rate=5e-5)
    # main(dev, filename, num_layers=14, learning_rate=1e-4)

    for num_ep in n_epoch:
        for lr in learning_rates:
            for n_layers in layers:
                tmp = {"INFO": {
                        "learning_rate": lr,
                        "num_layers": n_layers,
                        "num_epochs": num_ep
                    },
                    "RESULTS": main(dev, filename, num_layers=n_layers, learning_rate=lr, n_epochs=num_ep)
                }
                res.append(tmp)

    with open(out_filename, 'w') as file:
        json.dump(res, file)
