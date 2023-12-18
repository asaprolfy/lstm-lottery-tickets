# import time
import copy
import json

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.utils.prune as prune

wgt_mods = [
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LSTMCell,
    nn.GRUCell,
    nn.LSTM,
    nn.GRU
]


def create_apply_mask(model: nn.Module, prune_rate=0.1):
    mask = build_empty_mask_simple(model)

    new_mask = {}

    orig_wgt_tensors = {}

    i = 0

    for name, param in model.named_parameters():
        # if name in {"weight"}:
        if 'weight' in name:
            # wgt_tensor = param.data.numpy()
            wgt_dev = param.device
            wgt_tensor = param.data.cpu().numpy()
            orig_wgt_tensors[name] = wgt_tensor
            active = wgt_tensor[np.nonzero(wgt_tensor)]
            wgt_pcnt = np.percentile(np.abs(active), prune_rate * 100)
            print(f"wgt_pcnt:  {wgt_pcnt}")
            print(f"mask[name]:   {mask[name]}")
            wgt_mask = np.where(np.abs(wgt_tensor) < wgt_pcnt, 0, mask[name])
            param.data = torch.from_numpy(wgt_tensor * wgt_mask).to(wgt_dev)
            mask[name] = wgt_mask
            new_mask[name] = wgt_mask
            i += 1
    return new_mask, orig_wgt_tensors


def create_mask_simple(model: nn.Module, prune_rate=0.1):
    mask = build_empty_mask_simple(model)

    new_mask = {}

    orig_wgt_tensors = {}

    i = 0

    for name, param in model.named_parameters():
        # if name in {"weight"}:
        if 'weight' in name:
            # wgt_tensor = param.data.numpy()
            wgt_dev = param.device
            wgt_tensor = param.data.cpu().numpy()
            orig_wgt_tensors[name] = wgt_tensor
            active = wgt_tensor[np.nonzero(wgt_tensor)]
            wgt_pcnt = np.percentile(np.abs(active), prune_rate * 100)
            print(f"wgt_pcnt:  {wgt_pcnt}")
            print(f"mask[name]:   {mask[name]}")
            wgt_mask = np.where(np.abs(wgt_tensor) < wgt_pcnt, 0, mask[name])
            # param.data = torch.from_numpy(wgt_tensor * wgt_mask).to(wgt_dev)
            mask[name] = wgt_mask
            new_mask[name] = wgt_mask
            i += 1

    return new_mask, orig_wgt_tensors


def apply_mask_simple(model: nn.Module, mask):
    for name, param in model.named_parameters():
        # if name in {"weight"}:
        if 'weight' in name:
            wgt_dev = param.device
            # wgt_tensor = param.data.numpy()
            wgt_tensor = param.data.cpu().numpy()
            param.data = torch.from_numpy(wgt_tensor * mask[name]).to(wgt_dev)


def create_mask_dict(model: nn.Module, prune_rate=0.1):
    total_nonzero = 0
    mask = {}
    orig_wgt_tensors = {}
    for layer_name, layer in model.named_children():
        if type(layer) in wgt_mods:
            if str(layer_name) not in mask:
                mask[str(layer_name)] = {}
            if str(layer_name) not in orig_wgt_tensors:
                orig_wgt_tensors[str(layer_name)] = {}
            for param_name, param in layer.named_parameters():
                if 'weight' in param_name:
                    wgt_tensor = param.data.cpu().numpy()
                    orig_wgt_tensors[str(layer_name)][str(param_name)] = wgt_tensor
                    active = wgt_tensor[np.nonzero(wgt_tensor)]
                    total_nonzero += len(active)
                    wgt_pcnt = np.percentile(np.abs(active), prune_rate * 100)
                    print(f"wgt_pcnt:  {wgt_pcnt}")
                    wgt_mask = np.where(np.abs(wgt_tensor) < wgt_pcnt, 0, np.ones_like(wgt_tensor))
                    mask[str(layer_name)][str(param_name)] = wgt_mask

    return mask, orig_wgt_tensors, total_nonzero


def apply_mask_dict(model: nn.Module, mask):
    for layer_name, layer in model.named_children():
        if type(layer) in wgt_mods:
            for param_name, param in layer.named_parameters():
                if 'weight' in param_name:
                    wgt_dev = param.device
                    wgt_tensor = param.data.cpu().numpy()
                    param.data = torch.from_numpy(wgt_tensor * mask[str(layer_name)][str(param_name)]).to(wgt_dev)


def create_apply_mask_dict(model: nn.Module, prune_rate=0.1):
    total_nonzero = 0
    mask = {}
    orig_wgt_tensors = {}
    for layer_name, layer in model.named_children():
        if type(layer) in wgt_mods:
            if str(layer_name) not in mask:
                mask[str(layer_name)] = {}
            if str(layer_name) not in orig_wgt_tensors:
                orig_wgt_tensors[str(layer_name)] = {}
            for param_name, param in layer.named_parameters():
                if 'weight' in param_name:
                    wgt_dev = param.device
                    wgt_tensor = param.data.cpu().numpy()
                    orig_wgt_tensors[str(layer_name)][str(param_name)] = wgt_tensor
                    active = wgt_tensor[np.nonzero(wgt_tensor)]
                    total_nonzero += len(active)
                    wgt_pcnt = np.percentile(np.abs(active), prune_rate * 100)
                    print(f"wgt_pcnt:  {wgt_pcnt}")
                    wgt_mask = np.where(np.abs(wgt_tensor) < wgt_pcnt, 0, np.ones_like(wgt_tensor))
                    param.data = torch.from_numpy(wgt_tensor * wgt_mask).to(wgt_dev)
                    mask[str(layer_name)][str(param_name)] = wgt_mask

    return mask, orig_wgt_tensors, total_nonzero


def build_empty_mask_simple(model: nn.Module):
    i = 0
    mask = {}

    # for name, param in model.named_parameters():
    #     if 'weight' in name:
    #         i += 1
    #
    # mask = [None] * i
    #
    # i = 0

    # for layer in model.modules():
    #     if type(layer in wgt_mods):
    #         for name, param in layer.named_parameters():
    #             if 'weight' in name:
    #                 wgt_tensor = param.data.cpu().numpy()
    #                 mask[i] = np.ones_like(wgt_tensor)

    for name, param in model.named_parameters():
        if 'weight' in name:
            wgt_tensor = param.data.cpu().numpy()
            mask[name] = np.ones_like(wgt_tensor)

    return mask


def apply_mask(model: nn.Module, mask):
    i = 0

    for layer in model.modules():
        if type(layer) in wgt_mods:
            # weight = layer.weight
            # print(f"weight for layer: {type(layer)}  ||  weight:  {weight}")
            print(f"weight for layer: {type(layer)}")
            for name, param in layer.named_parameters():
                # if name in {"weight"}:
                if 'weight' in name:
                    wgt_dev = param.device
                    # wgt_tensor = param.data.numpy()
                    wgt_tensor = param.data.cpu().numpy()
                    param.data = torch.from_numpy(wgt_tensor * mask[i]).to(wgt_dev)
                    i += 1


def create_mask(model: nn.Module, prune_rate=0.1):
    mask = build_empty_mask(model)

    orig_wgt_tensors = {}

    i = 0

    for layer in model.modules():
        if type(layer) in wgt_mods:
            # weight = layer.weight
            # print(f"weight for layer: {type(layer)}  ||  weight:  {weight}")
            print(f"weight for layer: {type(layer)}")
            for name, param in layer.named_parameters():
                # if name in {"weight"}:
                if 'weight' in name:
                    # wgt_tensor = param.data.numpy()
                    wgt_dev = param.device
                    wgt_tensor = param.data.cpu().numpy()
                    orig_wgt_tensors[i] = wgt_tensor
                    active = wgt_tensor[np.nonzero(wgt_tensor)]
                    wgt_pcnt = np.percentile(np.abs(active), prune_rate * 100)
                    wgt_mask = np.where(np.abs(wgt_tensor) < wgt_pcnt, 0, mask[i])
                    # param.data = torch.from_numpy(wgt_tensor * wgt_mask).to(wgt_dev)
                    mask[i] = wgt_mask
                    i += 1
    return mask, orig_wgt_tensors


def build_empty_mask(model: nn.Module):
    i = 0
    # mask = []

    for layer in model.modules():
        if type(layer) in wgt_mods:
            for name, param in layer.named_parameters():
                # if name in {"weight"}:
                if 'weight' in name:
                    # wgt_tensor = param.data.numpy()
                    # wgt_dev = param.device
                    # wgt_tensor = param.data.cpu().numpy()
                    # mask.append(np.ones_like(wgt_tensor))
                    i += 1
    mask = [None] * i

    i = 0

    for layer in model.modules():
        if type(layer in wgt_mods):
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    wgt_tensor = param.data.cpu().numpy()
                    mask[i] = np.ones_like(wgt_tensor)

    return mask

