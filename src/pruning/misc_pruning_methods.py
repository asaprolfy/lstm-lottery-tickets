import logging

import torch
import torch.nn as nn
from torch.nn.utils import prune


wgt_mods = [
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LSTMCell,
    nn.GRUCell
]


class WeightThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold


def apply_weight_threshold_prune(module, threshold):
    WeightThresholdPruning.apply(module, threshold)
    return module


def apply_threshold_prune_all_layers(model: nn.Module, threshold, layer_names=None, threshold_list=None):
    params = [
        (module, "weight") for module in filter(lambda m: type(m) in wgt_mods, model.modules())
    ]

    prune.global_unstructured(params, pruning_method=WeightThresholdPruning, threshold=threshold)


def apply_l1un_prune_all_layers(model: nn.Module, rate):
    params = [
        (module, "weight") for module in filter(lambda m: type(m) in wgt_mods, model.modules())
    ]

    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=rate)


if __name__ == '__main__':
    print('Not a main file')
