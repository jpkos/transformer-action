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
#%%
# class Normalize:
#     def __call__(self, sample):
        
        
class VideoClipDataset(Dataset):
    
    def __init__(self, clip_dir, clip_length=5, fixed_transforms=None, random_transforms=True):
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
        self.clip_dir = clip_dir
        self.clips = glob.glob(os.path.join(clip_dir, '*.jpeg'))
        self.fixed_transforms= fixed_transforms
        self.clip_length = clip_length
        self.action_map = {'piercing':1, 'not':0, 'pierce':1, 'transport':0, 'nothing':0}
        
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        clip_path = self.clips[idx]
        action = os.path.basename(clip_path).split('_')[0]
        # print(action, self.action_map[action])
        clip = io.imread(clip_path)
        h,w,ch = clip.shape
        clip = clip.reshape(-1, 100, 100, ch)
        normalized = []
        if self.random_transforms:
            angle = int(np.random.choice([0, 90, 180, 270]))
            hue = (np.random.random()-0.5)*0.3
            hp = np.random.random()
            vp = np.random.random()
            prot = np.random.random()
            phue = np.random.random()
        if self.fixed_transforms:
            for i in range(self.clip_length):
                transformed = self.fixed_transforms(clip[i,:,:,:])
                if self.random_transforms:
                    if prot>0.3:
                        transformed = tf.rotate(transformed, angle)
                    if phue>0.3:
                        transformed = tf.adjust_hue(transformed, hue)
                    if hp>0.5:
                        transformed = tf.hflip(transformed)
                    if vp>0.5:
                        transformed = tf.hflip(transformed)
                
                normalized.append(transformed)
        clip = torch.stack(normalized)
        # plt.imshow(clip[0,:,:,:].permute(1,2,0))
        sample = {'clip':torch.stack(normalized), 'action':self.action_map[action]}
        return sample
#%%