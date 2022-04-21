# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:14:23 2022

@author: jankos
"""
import numpy as np
import cv2
import os
import glob
import torch
from torchvision import datasets, models, transforms
from models.transformer_yolov5 import BBTransformer
from utils.dataset import VideoClipDataset
#%%
model = BBTransformer(img_size=100,d_embed=768, n_head=12,
                        n_layers=12, n_classes=4, d_hid=3072)
model.load_state_dict(torch.load("local_files/toy_run29/w_best.pt"))
model.eval()
#%%
data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
#%%
data = 'local_files/toy_dataset/test'
clip_length = 12
labels = ['left', 'right', 'up', 'down']
batch_size = 1
n_work = 1
val_set = VideoClipDataset(data, clip_length=clip_length,
                             fixed_transforms=data_transforms,
                             random_transforms=False, labels=labels)
# val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,
#                                                           shuffle=True,
#                                                           num_workers=n_work)
#%%
action_labels = {0:'left', 1:'right', 2:'up', 3:'down'}
#%%
videoname='toy_action'
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')#VideoWriter_fourcc('m', 'p', '4', 'v')
video_writer = cv2.VideoWriter(f'local_files/{videoname}_1.mp4', fourcc,15, (100, 100))
start_frame = 1
i=start_frame
for i, d in enumerate(val_set):
    clip, action = d['clip'], d['action']
    output = model(clip.unsqueeze(0))
    _, pred = torch.max(output,1)
    pred = pred.item()
    for j in range(12):
        im = clip[j,:,:,:].squeeze()
        print(im.shape)
        im1 = im.permute(1,2,0)
        im1 = ((im1 + im1.min())/im1.max())*256
        im1 = im1.numpy().astype(np.uint8)
        im1 = cv2.putText(im1, f"{action_labels[pred]}", (35,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, 1)
        video_writer.write(im1)
video_writer.release()
cv2.destroyAllWindows()