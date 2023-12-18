# import time
import copy
import json

import torch
import torch.nn as nn
# import torch.nn.utils.prune as prune

from datasets.nlp.imdb import import_dataset

from train.train import train
from evaluate.test import evaluate
from models.sentiment_lstm_bidi_layers import BRNN_LSTM
from pruning.misc_pruning_methods import apply_threshold_prune_all_layers as prune_all
from pruning.misc_pruning_methods import apply_l1un_prune_all_layers as prune_all_l1un

from pruning.layerwise_weight_prune import create_mask, apply_mask


pruning_rates = [
    0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99,
    0.999, 0.9999, 0.99999
]

# pruning_rates = [x / 100 for x in range(5, 99)]
# pruning_rates.append(0.995)
# pruning_rates.append(0.999)


def main(device, run_configs, batch_size=32, hidden_dim=32):

    train_dl, test_dl, index_word_lex, word_index_lex = import_dataset(seq_length=batch_size)

    print(f"len train_dl:  {len(train_dl.dataset)}")
    print(f"train_dl head:  {train_dl.dataset[0]}")

    res = []

    for run in run_configs:

        # model = Sentiment_LSTM(batch_size, word_index_lex)
        model = BRNN_LSTM(batch_size, word_index_lex, num_layers=run['layers'], hidden_dim=hidden_dim)
        model.to(device)

        original_state_dict = copy.deepcopy(model.state_dict())

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=run['lr'])

        losses, accuracies = train(model, criterion, optimizer, train_dl, device, num_epochs=run['epochs'])

        torch.save(model,
                   f"./saves/brnn_lstm/imdb/initial_state_dict_{run['lr']}_{run['layers']}_{run['epochs']}.pth.tar")

        print(f"NUM LAYERS:  {run['layers']}  |  NUM EPOCHS:  {run['epochs']}  |  LEARN RATE:  {run['lr']}")

        print('Non pruned run:')
        evaluate(model, test_dl, device)

        acc = {}
        # for p_rate in pruning_rates:
        #     model_p = copy.deepcopy(model)
        #     prune_all_l1un(model_p, p_rate)
        #     print(f"{p_rate} prune test run: ")
        #     acc[p_rate] = evaluate(model_p, test_dl, device)

        for p_rate in pruning_rates:
            model_p = copy.deepcopy(model)
            mask = create_mask(model_p)
            apply_mask(model_p, mask)
            print(f"{p_rate} prune test run: ")
            acc[p_rate] = evaluate(model_p, test_dl, device)

        tmp = {"INFO": {
            "learning_rate": run["lr"],
            "num_layers": run["layers"],
            "num_epochs": run["epochs"]
        },
            "RESULTS": acc
        }
        res.append(tmp)
        del model_p

    return res


if __name__ == '__main__':

    out_filename = "./out/imdb-mask-results-0.json"

    runs = [
        {
            "layers": 2,
            "lr": 1e-5,
            "epochs": 50
        },
        {
            "layers": 4,
            "lr": 15e-6,
            "epochs": 50
        }
    ]

    dev = torch.device("mps")

    ress = main(dev, runs)

    with open(out_filename, 'w') as file:
        json.dump(ress, file)
