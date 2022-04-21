# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:55:52 2022

@author: jankos

For testing random stuff
"""

import numpy as np
import pandas as pd
import os
import glob
import cv2
import torch
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as tf
import torch
from utils.dataset import VideoClipDataset
from torchvision import datasets, models, transforms
from skimage import io, transform
from models.transformer_yolov5 import BBTransformer
#%%
class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = np.random.choice(self.angles)
        return tf.rotate(x, angle)

# rotation_transform = MyRotationTransform(angles=[-30, -15, 0, 15, 30])

#%%
torch.manual_seed(10)
data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # MyRotationTransform([90, 180, 270])
        # transforms.RandomRotation((0,270))
    ])
train_set = VideoClipDataset('local_files/toy_dataset/val', clip_length=12,
                             fixed_transforms=data_transforms, random_transforms=False, labels=['left', 'right', 'up', 'down'])
# train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,
#                                                          shuffle=False,
#                                                          num_workers=1,
#                                                          pin_memory=True)

#%%
d = train_set[5]
#%%
for i in range(20,25):
    fig, ax = plt.subplots(ncols=12, figsize=(5,15))
    axes = ax.flatten()
    clip1 = train_set[i]
    im = clip1['clip']
    stack = []
    for i in range(12):
        im0 = im[i,:,:,:]
        im1 = im0.permute(1,2,0)
        # im1 = im1 - im1.min()
        # im1 = (im1)
        # im1 = cv2.cvtColor(im1.numpy(), cv2.COLOR_RGB2BGR)
        axes[i].imshow(im1)
#%%
bbt = BBTransformer(100, 96, 8, 8, 4)
bbt.eval()
#%%
y = bbt(im.unsqueeze(0))
#%%
clip_path = 'local_files/pierce_full/train/not_piercing_Patient__2020-10-22-10-10-11-538-000_2621.jpeg'
clip = io.imread(clip_path)
#%%
h,w,ch = clip.shape
clip = clip.reshape(-1, 100, 100, ch)
clip = clip.transpose(0,3,1,2)
# clip = clip.reshape(12, -1, w, ch)
#%%
clip = clip.transpose(3,0,1,2)
clip = transform.resize(clip, (ch, 12, 100, 100))

clip = torch.from_numpy(clip).float()
clip = clip.permute(1,0,2,3)

clip2 = clip[0,:,:,:]
clip2 = clip2.permute(1,2,0)

cv2.imshow('t', clip2.numpy())
#%%
df = pd.read_csv('local_files/pierce_full_weights/sixth_run/results.csv')
df.plot('epoch', 'accuracy')