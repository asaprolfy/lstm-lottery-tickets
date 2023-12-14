import time

import numpy as np

import torch
import torch.nn as nn


def evaluate(model, test_dl, device):
    loss, correct_preds, total_samples = 0, 0, 0

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            data = batch[0].to(device)
            target = batch[1].to(device)
            h0, c0 = model.init_hidden()
            h0 = h0.to(device)
            c0 = c0.to(device)
            out, hidden = model(data, (h0, c0))
            _, predicted = torch.max(out, 1)
            correct_preds += (predicted == target).sum().item()
            total_samples += target.size(0)

    test_accuracy = correct_preds / total_samples
    average_test_loss = loss / (i + 1)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}% | Average Test Loss: {average_test_loss}')

    model.train()


if __name__ == '__main__':
    print('Not a main file')
