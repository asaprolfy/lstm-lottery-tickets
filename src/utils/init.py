import torch
import torch.nn as nn
import torch.nn.init as init

BN_WEIGHT_MEAN = 1
BN_WEIGHT_STDDEV = 0.02


def init_weights(module):
    print("init weights for layer: ")
    print(module)

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
        case nn.Conv1d:
            init.normal_(module.weight.data)
            if module.bias:
                init.normal_(module.bias.data)
        case nn.Conv2d:
            init.xavier_normal_(module.weight.data)
            if module.bias:
                init.normal_(module.bias.data)
        case nn.Conv3d:
            init.xavier_normal_(module.weight.data)
            if module.bias:
                init.normal_(module.bias.data)
        case nn.ConvTranspose1d:
            init.normal_(module.weight.data)
            if module.bias:
                init.normal_(module.bias.data)
        case nn.ConvTranspose2d:
            init.xavier_normal_(module.weight.data)
            if module.bias:
                init.normal_(module.bias.data)
        case nn.ConvTranspose3d:
            init.xavier_normal_(module.weight.data)
            if module.bias:
                init.normal_(module.bias.data)
        case nn.BatchNorm1d:
            init.normal_(module.weight.data, mean=BN_WEIGHT_MEAN, std=BN_WEIGHT_STDDEV)
            init.zeros_(module.bias.data)
        case nn.BatchNorm2d:
            init.normal_(module.weight.data, mean=BN_WEIGHT_MEAN, std=BN_WEIGHT_STDDEV)
            init.zeros_(module.bias.data)
        case nn.BatchNorm3d:
            init.normal_(module.weight.data, mean=BN_WEIGHT_MEAN, std=BN_WEIGHT_STDDEV)
            init.zeros_(module.bias.data)
        case nn.GRU:
            for p in module.parameters():
                if len(p.shape) >= 2:
                    init.orthogonal_(p.data)
                else:
                    init.normal_(p.data)
        case nn.GRUCell:
            for p in module.parameters():
                if len(p.shape) >= 2:
                    init.orthogonal_(p.data)
                else:
                    init.normal_(p.data)
