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

#%%
# class Normalize:
#     def __call__(self, sample):
        
        
class VideoClipDataset(Dataset):
    
    def __init__(self, clip_dir, clip_length=5, transform=None, random_transforms=True):
        """
        For reading training clips from a directory. Images are assumed to be
        stacked vertically so that shape of clip: (clip_length*h, w, ch) where
        h = height, w = width, ch = channels.
        clip_dir : str
            directory with video_clips
        clip_length : int
            how many frames in a clip The default is 5.
        transform : torch transform, optional
            custom pytorch transformations applied. The default is None.

        Returns
        -------
        None.

        """
        self.random_transforms = random_transforms
        self.clip_dir = clip_dir
        self.clips = glob.glob(os.path.join(clip_dir, '*.jpeg'))
        self.transform = transform
        self.clip_length = clip_length
        #self.action_map = {'Drilling':0, 'Not drilling':1}
        #self.carry_map = {'Drilling':0, 'Not drilling':1}
        # self.action_map = {'move':0, 'hold still':1, 'grab':2,
        #                    'transport':3, 'NV':4, 'idle':5}
        # self.carry_map = {'needle':0, 'thread':1, 'nothing':2}
        # self.action_map = {'pierce':0, 'transport':1, 'nothing':2}
        # self.carry_map = {'pierce':0, 'transport':1, 'nothing':2}
        self.action_map = {'piercing':1, 'not':0, 'pierce':1, 'transport':0, 'nothing':0}
        self.carry_map = {'piercing':1, 'not':0, 'pierce':1, 'transport':0, 'nothing':0}
        
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        clip_path = self.clips[idx]
        action = os.path.basename(clip_path).split('_')[0]
        carrying = os.path.basename(clip_path).split('_')[0]
        clip = io.imread(clip_path)
        h,w,ch = clip.shape
        clip = clip.reshape(self.clip_length, -1, w, ch)
        clip = clip.transpose(3,0,1,2)
        clip = transform.resize(clip, (ch, self.clip_length, 100, 100))
        clip = torch.from_numpy(clip).float()
        clip = clip.permute(1,0,2,3)
        sample = {'clip':clip, 'action':self.action_map[action], 'carrying':self.carry_map[carrying]}
        normalized = []
        # print(clip.shape)
        if self.random_transforms:
            angle = int(np.random.choice([0, 45, 90, 135, 225, 315]))
            hue = (np.random.random()-0.5)*0.4
            hp = np.random.random()
            vp = np.random.random()
            prot = np.random.random()
            phue = np.random.random()
        if self.transform:
            for i in range(self.clip_length):
                transformed = self.transform(tf.to_pil_image(sample['clip'][i,:,:,:]))
                # print(self.transform.get_params())
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
            # normalized.append(self.transform(tf.to_pil_image(sample['clip'])))
                
        sample['clip'] = torch.stack(normalized)
        return sample
#%%