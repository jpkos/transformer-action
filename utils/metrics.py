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
from sklearn.metrics import confusion_matrix

#%%
class MetricsTracker():
    
    def __init__(self):
        self.results = []
        self.best = 0
        self.columns = ['accuracy', 'precision', 'recall', 'f1']
        
    def calculate(self, true, pred):
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        accuracy = (tp + tn)/(tp+tn+fp+fn)
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        f1 = 2*(precision*recall)/(precision + recall)
        if accuracy>self.best:
            self.best = accuracy
        self.results.append([accuracy, precision, recall, f1])
        
    def as_df(self):
        return pd.DataFrame(self.results, columns=self.columns)