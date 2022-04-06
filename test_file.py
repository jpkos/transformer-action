# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:55:52 2022

@author: jankos
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
from dataset import VideoClipDataset
from torchvision import datasets, models, transforms
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
train_set = VideoClipDataset('pierce_full/train/', clip_length=12,
                             transform=data_transforms, random_transforms=False)
# train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,
#                                                          shuffle=False,
#                                                          num_workers=1,
#                                                          pin_memory=True)
#%%
for i in range(160,162):
    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(5,15))
    axes = ax.flatten()
    clip1 = train_set[i]
    im = clip1['clip']
    stack = []
    for i in range(12):
        im0 = im[i,:,:,:]
        im1 = im0.permute(1,2,0)
        im1 = im1 - im1.min()
        im1 = (im1/im1.max())
        # im1 = cv2.cvtColor(im1.numpy(), cv2.COLOR_RGB2BGR)
        axes[i].imshow(im1)

