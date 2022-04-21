# -*- coding: utf-8 -*-

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
import pandas as pd
from shutil import copyfile
import os
from pathlib import Path
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report

#%%
class MetricsTracker():
    
    def __init__(self, labels=[]):
        self.results = []
        self.best = 0
        self.columns = ['epoch', 'accuracy', 'precision', 'recall', 'f1', 'mode']
        
    def calculate(self, true, pred, mode, epoch=0, prints=True):
        report = classification_report(true, pred, output_dict=True, zero_division=0)
        accuracy = report['accuracy']
        macro_dic = report['macro avg']
        precision = macro_dic['precision']
        recall = macro_dic['recall']
        f1 = macro_dic['f1-score']
        if accuracy>self.best and mode=='val':
            self.best = accuracy
        self.results.append([epoch, accuracy, precision, recall, f1, mode])
        if prints:
            print(f'mode: {mode}, epoch: {epoch}, accuracy: {accuracy:.2f}, precision: {precision:.2f} ' + 
                  f'recall: {recall:.2f}, f1: {f1:.2f}')
        
    def as_df(self):
        return pd.DataFrame(self.results, columns=self.columns)