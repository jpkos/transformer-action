# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:13:09 2022

@author: jankos
"""
import yaml
import torch.nn as nn
import torch.optim as optim
from utils.dataset import VideoClipDataset
import torch
import os


def setup_train(yaml_path, model):
    with open(yaml_path, 'r') as train_params:
        tp = yaml.safe_load(train_params)
    if tp['criterion'] == 'crossentropy':
        tp['criterion'] = nn.CrossEntropyLoss(label_smoothing=tp['label_smoothing'])
    else:
        tp['criterion'] = None
        raise ValueError("Not valid criterion, only 'crossentropy' supported for now.")
        # print("Not valid criterion, only 'crossentropy' supported for now.")
    if tp['optimizer'] == 'adam':
        tp['optimizer'] = optim.Adam(model.parameters(), weight_decay=tp['lr_gamma'], lr=tp['lr'])
    elif tp['optimizer'] == 'sgd':
        tp['optimizer'] = optim.SGD(model.parameters(), lr=tp['lr'], weight_decay=tp['lr_gamma'])
    elif tp['optimizer'] == 'adamw':
        tp['optimizer'] == optim.AdamW(model.parameters, lr=tp['lr'], weight_decay=tp['lr_gamma'])
    else:
        tp['optimizer'] = None
        raise ValueError("Not valid loss, try 'adam', 'adamw' or 'sgd'")
        # print("Not valid loss, try 'adam' or 'sgd'")
    print("Training parameters: ", tp)
    return tp

def setup_model(yaml_path):
    with open(yaml_path, 'r') as model_params:
        mp = yaml.safe_load(model_params)
    print("Model parameters: ", mp)
    return mp

def setup_dataloaders(data, clip_length, data_transforms, batch_size, n_work, labels):
    train_set = VideoClipDataset(data[0], clip_length=clip_length,
                                 fixed_transforms=data_transforms,
                                 random_transforms=False, labels=labels)
    print(f'Found {len(train_set)} files in {data[0]}')
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,
                                                             shuffle=True,
                                                             num_workers=n_work,
                                                             pin_memory=True)
    val_set = VideoClipDataset(data[1], clip_length=clip_length,
                                 fixed_transforms=data_transforms,
                                 random_transforms=False, labels=labels)
    print(f'Found {len(val_set)} files in {data[1]}')
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,
                                                             shuffle=False,
                                                             num_workers=n_work,
                                                             pin_memory=True)
    return {'train':train_loader, 'val':val_loader}
    