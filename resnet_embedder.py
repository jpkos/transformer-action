# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 16:04:19 2021

@author: jankos
"""


# import cv2
# import numpy as np
from __future__ import print_function
from __future__ import division
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
import torchvision
import torch
from torchvision import datasets, models, transforms

import time
import os
import copy
#%%Train func
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

if __name__=='__main__':
    #%%params
    data_dir = 'hymenoptera_data/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Batch size for training (change depending on how much memory you have)
    batch_size = 16
    # Number of epochs to train for
    num_epochs = 30
    
    input_size = 224
    
    #%%Resnet embedder
    embed_dim = 2
    embed_model = models.resnet18(pretrained=True)
    embed_model.fc = nn.Linear(512, embed_dim)
    #%%Optimizer and stuff
    params_to_update = embed_model.parameters()
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    #%%Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    #%%
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    
    #%%Load to gpu
    embed_model = embed_model.to(device)
    #%%Loss
    criterion = nn.CrossEntropyLoss()
    print(f'DEVICE IS {device}')
    embed_model, hist = train_model(embed_model, dataloaders_dict, criterion, optimizer_ft,
                                 num_epochs=num_epochs)
    torch.save(embed_model.state_dict(), 'embed_test.pt')