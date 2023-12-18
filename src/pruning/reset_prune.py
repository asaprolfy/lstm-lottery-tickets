import numpy as np

import torch
import torch.nn as nn


wgt_mods = [
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LSTM,
    nn.GRU,
    nn.LSTMCell,
    nn.GRUCell
]


def reset_model(model: nn.Module, original_state_dict):
    i = 0
    for layer in model.modules():
        if type(layer) in wgt_mods:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    wgt_dev = param.device
                    param.data = torch.from_numpy(original_state_dict[name].cpu().numpy()).to(wgt_dev)
                    i += 1
                if 'bias' in name:
                    param.data = original_state_dict[name]


def reset_non_mask(model: nn.Module, mask, original_state_dict):
    i = 0
    for layer in model.modules():
        if type(layer) in wgt_mods:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    wgt_dev = param.device
                    param.data = torch.from_numpy(mask[i] * original_state_dict[layer][name].cpu().numpy()).to(wgt_dev)
                    i += 1
                if 'bias' in name:
                    param.data = original_state_dict[name]


def reset_orig_wgt_tensors(model: nn.Module, mask, orig_wgt_tensors, original_state_dict):
    i = 0
    for layer in model.modules():
        if type(layer) in wgt_mods:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    wgt_dev = param.device
                    param.data = torch.from_numpy(mask[name] * orig_wgt_tensors[name]).to(wgt_dev)
                    i += 1
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.data = original_state_dict[name]
