# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:18:51 2021

@author: jankos

For training the Bounding Box Transformer
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
from utils.setup_funcs import setup_train, setup_model, setup_dataloaders
from utils.general import create_save_folder
import pandas as pd
from shutil import copyfile
import os
from pathlib import Path
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
from utils.metrics import MetricsTracker
import numpy as np
import matplotlib.pyplot as plt
#%%
torch.manual_seed(0)
#%%
def train(model, optimizer, criterion, scheduler, dataloaders, epochs, device, save_folder, n_save):
    """
    Standard Pytorch training loop. Saves accuracy and loss into csv file
    after each epoch

    """
    # results = []
    best_accuracy = 0
    mtrack = MetricsTracker()
    for epoch in range(epochs):
        print(f'epoch {epoch}')
        for mode in ['train', 'val']:
            corrects = 0
            loss_av = 0
            dets = 0
            start_time = time.time()
            predictions = []
            trues = []
            if mode == 'train':
                model.train()
            else:
                model.eval()
            for i, data in enumerate(dataloaders[mode]):
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
                predictions.append(pred)
                trues.append(action)
                corrects += torch.sum(pred == action)
                loss_av += loss
                dets += len(pred)
                print(f'Predicted imgs: {dets}', end='\r')
            print(f'{mode}: elapsed time: {time.time() - start_time}')
            print(f'correct {corrects}/{dets}')
            print(f'ratio {corrects/dets}')
            print(f'{loss_av/dets}')
            if mode == 'train':
                scheduler.step()
            if mode == 'val':
                predictions = torch.cat(predictions).cpu().numpy()
                trues = torch.cat(trues).cpu().numpy()
                mtrack.calculate(trues, predictions)
                if mtrack.best>best_accuracy:
                    best_accuracy = mtrack.best
                    torch.save(model.state_dict(), f'{save_folder}/w_best.pt')
            if epoch%n_save==0:
                torch.save(model.state_dict(), f'{save_folder}/w_{epoch}.pt')
        results_df = mtrack.as_df()
        results_df = results_df.assign(epoch=np.arange(1,len(results_df)+1))
        results_df.to_csv(f'{save_folder}/results.csv', index=None)
    results_df.plot('epoch',['accuracy', 'precision', 'recall', 'f1'], 
                    subplots=True, layout=(2,2), figsize=(10,10))
    plt.savefig(f'{save_folder}/results.png')
    torch.save(model.state_dict(), f'{save_folder}/w_last.pt')
#%%
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='path to yaml file for training')
    parser.add_argument('--model', type=str, help='path to yaml file for model')
    parser.add_argument('--save_folder', type=str, help='where to save the weights')
    parser.add_argument('--n_save', type=int, default=5, help='save weights after every n epochs')
    args = parser.parse_args()
    #Create save folder and save training attributes
    # Path(args.save_folder).mkdir(parents=True, exist_ok=True)
    args.save_folder = create_save_folder(args.save_folder)
    copyfile(args.train, os.path.join(args.save_folder, os.path.basename(args.train)))
    copyfile(args.model, os.path.join(args.save_folder, os.path.basename(args.model)))
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
    #Find training parameters from Yaml file given with --train
    (batch_size,
     n_work,
     lr,
     epochs,
     criterion,
     optimizer,
     data,
     lr_step,
     lr_gamma,
     labels) = setup_train(args.train, bbt)
    #Setup transforms
    fixed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    #Setup dataloaders
    dataloaders = setup_dataloaders(data, clip_length, fixed_transforms, batch_size, n_work, labels)
    #Start training
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    train(bbt, optimizer, criterion, lr_scheduler, dataloaders, epochs, device,
          args.save_folder, args.n_save)
    
    

