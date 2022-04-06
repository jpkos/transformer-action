# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:18:51 2021

@author: jankos
"""

import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import math
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from transformer_yolov5 import BBTransformer, BoxEmbedder
from dataset import VideoClipDataset
import time
import yaml
import argparse
from utils.setup_funcs import setup_train, setup_model
#%%
# def setup_train(yaml_path):
#     with open(yaml_path, 'r') as train_params:
#         tp = yaml.safe_load(train_params)
#     batch_size = tp['batch_size']
#     n_work = tp['n_work']
#     lr = tp['lr']
#     epochs = tp['epochs']
#     if tp['criterion'] == 'crossentropy':
#         criterion = nn.CrossEntropyLoss()
#     else:
#         criterion = None
#         print("Not valid criterion, only 'crossentropy' supported for now.")
        
#     if tp['optimizer'] == 'adam':
#         optimizer = optim.Adam(bbt.parameters(), weight_decay=0.1)
#     elif tp['optimizer'] == 'sgd':
#         optimizer = optim.SGD(bbt.parameters(), lr=0.001)
#     else:
#         optimizer = None
#         print("Not valid loss, try 'adam' or 'sgd'")
        
#     return batch_size, n_work, lr, epochs, criterion, optimizer

# def setup_model
#%%
def train(model, optimizer, criterion, loaders, epochs, device):
    for epoch in range(epochs):
        print(f'epoch {epoch}')
        for mode in ['train', 'val']:
            corrects = 0
            loss_av = 0
            dets = 0
            start_time = time.time()
            if mode == 'train':
                model.train()
            else:
                model.eval()
            for i, data in enumerate(loaders[mode]):
                clip, action, carrying = data['clip'], data['action'], data['carrying']
                if clip.shape[0] == 1:
                    continue
                clip = clip.squeeze().to(device)
                detected = action
                detected = detected.to(device)
                optimizer.zero_grad()
                output = model(clip)
                _, pred = torch.max(output,1)
                loss = criterion(output, detected)
                loss.backward()
                optimizer.step()
                corrects += torch.sum(pred == detected)
                loss_av += loss
                dets += len(pred)
                print(f'Predicted imgs: {i+1}', end='\r')
            print(f'{mode}: elapsed time: {time.time() - start_time}')
            print(f'correct {corrects}/{dets}')
            print(f'ratio {corrects/dets}')
            print(f'{loss_av/dets}')
            if epoch%5==0:
                torch.save(model.state_dict(), f'pierce_full_weights/w_{epoch}.pt')
        torch.save(model.state_dict(), 'pierce_full_weights/w_last.pt')
#%%
data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='path to yaml file for training')
    parser.add_argument('--model', type=str, help='path to yaml file for model')
    args = parser.parse_args()
    
    n_layers, n_head, n_classes, d_embed, img_size, clip_length = setup_model(args.model)
    bbt = BBTransformer(img_size=img_size, d_embed=d_embed, n_head=n_head,
                        n_layers=n_layers, n_classes=n_classes)
    bbt = bbt.to(device)
    
    batch_size, n_work, lr, epochs, criterion, optimizer = setup_train(args.train, bbt)

    train_set = VideoClipDataset('pierce_full/train/', clip_length=clip_length,
                                 transform=data_transforms, random_transforms=True)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,
                                                             shuffle=False,
                                                             num_workers=n_work,
                                                             pin_memory=True)
    val_set = VideoClipDataset('pierce_full/val/', clip_length=clip_length,
                                 transform=data_transforms, random_transforms=False)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,
                                                             shuffle=False,
                                                             num_workers=n_work,
                                                             pin_memory=True)
    loaders = {'train':train_loader, 'val':val_loader}
    train(bbt, optimizer, criterion, loaders, epochs, device)
    
    

