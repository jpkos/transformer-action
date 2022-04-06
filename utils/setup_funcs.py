# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:13:09 2022

@author: jankos
"""
import yaml
import torch.nn as nn
import torch.optim as optim

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
        optimizer = optim.Adam(model.parameters(), weight_decay=0.1)
    elif tp['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001)
    else:
        optimizer = None
        print("Not valid loss, try 'adam' or 'sgd'")
    print("Training parameters: ", tp)
    return batch_size, n_work, lr, epochs, criterion, optimizer

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