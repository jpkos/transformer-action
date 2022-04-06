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
from models.transformer_yolov5 import BBTransformer, BoxEmbedder
# from dataset import VideoClipDataset
import time
import yaml
import argparse
from utils.setup_funcs import setup_train, setup_model, setup_loaders
import pandas as pd
#%%
def train(model, optimizer, criterion, loaders, epochs, device, save_folder):
    results = []
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
                optimizer.zero_grad(set_to_none=True)
                clip, action = data['clip'], data['action']
                if clip.shape[0] == 1:
                    continue
                clip = clip.squeeze().to(device)
                action = action.to(device)
                with torch.set_grad_enabled(mode == 'train'):
                    output = model(clip)
                    _, pred = torch.max(output,1)
                    loss = criterion(output, action)
                    # print(pred, action)
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                corrects += torch.sum(pred == action)
                loss_av += loss
                dets += len(pred)
                print(f'Predicted imgs: {i+1}', end='\r')
            print(f'{mode}: elapsed time: {time.time() - start_time}')
            print(f'correct {corrects}/{dets}')
            print(f'ratio {corrects/dets}')
            print(f'{loss_av/dets}')
            if mode == 'val':
                results.append([epoch+1, (corrects/dets).item(), (loss_av/dets).item()])
            if epoch%5==0:
                torch.save(model.state_dict(), f'{save_folder}/w_{epoch}.pt')
        results_df = pd.DataFrame(results, columns=['epoch', 'accuracy', 'loss'])
        results_df.to_csv(f'{save_folder}/results.csv', index=None)
        torch.save(model.state_dict(), f'{save_folder}/w_last.pt')
#%%
data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='path to yaml file for training')
    parser.add_argument('--model', type=str, help='path to yaml file for model')
    parser.add_argument('--save_folder', type=str, help='where to save the weights')
    args = parser.parse_args()
    #Find model parameters from Yaml file given with --model
    (n_layers,
     n_head,
     n_classes,
     d_embed,
     img_size,
     clip_length) = setup_model(args.model)
    #Load model
    bbt = BBTransformer(img_size=img_size, d_embed=d_embed, n_head=n_head,
                        n_layers=n_layers, n_classes=n_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bbt = bbt.to(device)
    #Find training parameters
    (batch_size,
     n_work,
     lr,
     epochs,
     criterion,
     optimizer,
     data) = setup_train(args.train, bbt)
    #Setup dataloaders
    loaders = setup_loaders(data, clip_length, data_transforms, batch_size, n_work)
    #Start training
    save_folder = 'local_files/pierce_full_weights/'
    train(bbt, optimizer, criterion, loaders, epochs, device, save_folder)
    
    

