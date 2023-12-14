import time

import numpy as np

import torch
import torch.nn as nn


def train(model, criterion, optimizer, train_dl,
          device, num_epochs=50):

    losses, accuracies = [], []

    model.train()

    for epoch in range(num_epochs):
        h0, c0 = model.init_hidden()
        h0, c0 = h0.to(device), c0.to(device)

        total_loss = 0
        total_samples = 0
        correct_preds = 0

        startt = time.time()

        for i, batch in enumerate(train_dl):
            data = batch[0].to(device)
            target = batch[1].to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                out, hidden = model(data, (h0, c0))
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, pred = torch.max(out, 1)
                correct_preds += (pred == target).sum().item()

                total_samples += target.size(0)

        endt = time.time()
        mins, secs = calc_time(startt, endt)
        losses.append(total_loss / (i + 1))
        accuracy = correct_preds / total_samples
        accuracies.append(accuracy)

        s = f"epoch: [{epoch}/{num_epochs}] "
        s += f"| epoch time: {mins}m {secs}s | loss: {losses[-1]:.3f} | accuracy: {accuracy * 100:.2f}"
        print(s)


def calc_time(startt, endt):
    dif = endt - startt
    mins = int(dif / 60)
    secs = dif - (mins * 60)
    return mins, secs


if __name__ == '__main__':
    print('Not a main file')
