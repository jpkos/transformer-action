# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:15:50 2021

@author: jankos
"""

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as tf
import glob
import time
import cv2
from utils.data_transforms import RandomTransforms
#%%
     
class VideoClipDataset(Dataset):
    
    def __init__(self, clip_dir, clip_length=5, fixed_transforms=None, random_transforms=None, labels=None):
        """
        For reading training clips from a directory. Images are assumed to be
        stacked vertically so that shape of clip: (clip_length*h, w, ch) where
        h = height, w = width, ch = color channels.
        clip_dir : str
            directory with video_clips
        clip_length : int
            how many frames in a clip The default is 5.
        fixed_fixed_transformss: torch transform, optional
            custom pytorch transformations applied. The default is None.

        Returns
        -------
        None.

        """
        self.random_transforms = random_transforms
        clip_data = pd.read_csv(os.path.join(clip_dir, 'labels.csv'))
        self.clips = clip_data['file'].values
        self.labels = clip_data['label'].values
        self.fixed_transforms = fixed_transforms
        self.clip_length = clip_length
        self.label_map = dict(zip(labels, [x for x in range(len(labels))]))
        
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        clip_path = self.clips[idx]
        label = self.labels[idx]
        clip = io.imread(clip_path)
        h,w,ch = clip.shape
        clip = clip.reshape(-1, 100, 100, ch)
        normalized = []

        if self.fixed_transforms:
            for i in range(self.clip_length):
                transformed = self.fixed_transforms(clip[i,:,:,:])
                if self.random_transforms:
                    rt = RandomTransforms()
                    transformed = rt(transformed)
                
                normalized.append(transformed)
        clip = torch.stack(normalized)
        sample = {'clip':torch.stack(normalized), 'action':self.label_map[label]}
        return sample
#%%