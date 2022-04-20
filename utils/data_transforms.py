# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as tf

class RandomTransforms(object):
    
    def __init__(self):
        self.angle = int(np.random.choice([0, 90, 180, 270]))
        self.hue = (np.random.random()-0.5)*0.3
        # self.hp = np.random.random()
        # self.vp = np.random.random()
        self.prot = np.random.random()
        self.phue = np.random.random()
    
    def __call__(self, x):
        if self.prot>0.3:
            x = tf.rotate(x, self.angle)
        if self.phue>0.3:
            x = tf.adjust_hue(x, self.hue)
        # if self.hp>0.5:
        #     x = tf.hflip(x)
        # if self.vp>0.5:
        #     x = tf.hflip(x)
        return x
    