#!/usr/bin/env python
import torch
from os.path import join
from .utilities import check_makedirs


def save_checkpoint(model, other_state={}, sav_path='', filename='model.pth.tar', stage=1):
    if isinstance(model, torch.nn.Module):
        weight = model.state_dict()
    else:
        raise ValueError('model must be nn.Module')
    check_makedirs(sav_path)

    if stage == 2: # remove vqvae part
        for k in list(weight.keys()):
            if 'autoencoder' in k:
                weight.pop(k)

    other_state['state_dict'] = weight
    filename = join(sav_path, filename)
    torch.save(other_state, filename)
