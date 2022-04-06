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
    batch_size = tp['batch_size']
    n_work = tp['n_work']
    lr = tp['lr']
    epochs = tp['epochs']
    if tp['criterion'] == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = None
        print("Not valid criterion, only 'crossentropy' supported for now.")
        
    if tp['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=0.1, lr=lr)
    elif tp['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = None
        print("Not valid loss, try 'adam' or 'sgd'")
    print("Training parameters: ", tp)
    data = tp['data']
    return batch_size, n_work, lr, epochs, criterion, optimizer, data

def setup_model(yaml_path):
    with open(yaml_path, 'r') as model_params:
        mp = yaml.safe_load(model_params)
    n_layers = mp['n_layers']
    n_head = mp['n_head']
    n_classes = mp['n_classes']
    d_embed = mp['d_embed']
    img_size = mp['img_size']
    clip_length = mp['clip_length']
    print("Model parameters: ", mp)
    return n_layers, n_head, n_classes, d_embed, img_size, clip_length

def setup_loaders(data, clip_length, data_transforms, batch_size, n_work):
    train_set = VideoClipDataset(data[0], clip_length=clip_length,
                                 fixed_transforms=data_transforms, random_transforms=False)
    print(f'Found {len(train_set)} files in {data[0]}')
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,
                                                             shuffle=True,
                                                             num_workers=n_work,
                                                             pin_memory=True)
    val_set = VideoClipDataset(data[1], clip_length=clip_length,
                                 fixed_transforms=data_transforms, random_transforms=False)
    print(f'Found {len(val_set)} files in {data[1]}')
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,
                                                             shuffle=False,
                                                             num_workers=n_work,
                                                             pin_memory=True)
    return {'train':train_loader, 'val':val_loader}
    