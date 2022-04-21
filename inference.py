# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:27:47 2021

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
from utils.dataset import VideoClipDataset
import time
import glob
import os
import numpy as np
import matplotlib.image as mpimg
import time
import pandas as pd
from skimage import io, transform
#%%
def detect(model, data_path, device):
    model.eval()
    files = glob.glob(os.path.join(data_path, '*.jpeg'))
    for i, file in enumerate(files):
        img = Image.open(file)
        img = transforms.ToPILImage(img)

#%%
# P6_path = r"C:\Users\jankos\tyokansio\projektit\carry-detect\P6clips_12f\val"
# data_path = 'action_dataset/val/'
#data_path = r"C:\Users\jankos\tyokansio\projektit\transformer-action\needle_stacks_for_testing2"
data_path = 'local_files/action_dataset/val/'
n_work = 1
batch_size = 16
data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
val_set = VideoClipDataset(data_path, clip_length=12,
                             fixed_transforms=data_transforms, labels=['needle', 'thread', 'nothing'])
val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,
                                                             shuffle=False,
                                                             num_workers=n_work,
                                                             pin_memory=True)

        

#%%
model = BBTransformer(img_size=100,d_embed=768, n_head=12,
                        n_layers=12, n_classes=4, d_hid=3072)
model.load_state_dict(torch.load('local_files/action_run10/w_last.pt'))
model.eval()
#%%
criterion = nn.CrossEntropyLoss()
#%%
corrects = 0
loss_av = 0
dets = 0
class_map =  {v: k for k, v in val_set.label_map.items()}
print(class_map)

class_map =  {v: k for k, v in val_set.label_map.items()}
print(class_map)
start_time = time.time()
predictions = []
true = []
fignums = []
for i, data in enumerate(val_set):
    clip = data['clip']
    clip = clip.unsqueeze(dim=0)
    out = model(clip)
    _, pred = torch.max(out, 1)
    target = torch.Tensor([data['action']]).long()
    loss = criterion(out, target)
    corrects += torch.sum(pred == target)
    loss_av += loss
    dets += len(pred)
    print(i)
    # print('val: elapsed time: {}'.format(time.time() - start_time))
    # print(f'correct {corrects}/{dets}')
    # print(f'{loss_av/dets}')
    predictions.append(class_map[pred.item()])
    true.append(class_map[data['action']])
    fignums.append(i)
    title_text = 'piercing\npredicted: {:>}\ntrue: {:>}'.format(class_map[pred.item()],
                                                                class_map[data['action']])
    im = cv2.imread(val_set.clips[i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    print(title_text)
    # im = np.hstack([im[0:600,:,:],im[600:,:,:]])
    plt.figure(figsize=(4,12))
    plt.imshow(im.reshape(600,200,3))#.transpose(1,0,2))
    plt.title(title_text, fontsize=20)
    plt.tight_layout()
    plt.savefig('local_files/action_run10/val/action_predict_{}.png'.format(i))
    plt.close()
    # if i>1:
        # break
df = pd.DataFrame({'predicted':predictions, 'true':true, 'fignum':fignums})
df.to_csv('pierce_P5_predicted.csv', index=None)