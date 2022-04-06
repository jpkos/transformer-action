# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:40:10 2021

@author: jankos
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
#%%
input_size = 224
#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
embed_dim = 2
embed_model = models.resnet18()
embed_model.fc = nn.Linear(512, embed_dim)
embed_model.load_state_dict(torch.load('embed_test.pt'))
embed_model = embed_model.to(device)
embed_model.eval()
#%%
data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
#%%
dataset = datasets.ImageFolder('hymenoptera_data/val/', data_transforms)
#%%

for img, label in dataset:
    # img, label = datasets[i]
    out = embed_model(torch.unsqueeze(img, 0).to(device))
    print(f'prediction {torch.max(out,1)[1].item()}, correct {label}')
