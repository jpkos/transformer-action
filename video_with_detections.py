# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 17:22:41 2022

@author: jankos
"""


import cv2
import pandas as pd
import numpy as np
import glob
import re
import os
from skimage import io, transform
import torchvision.transforms.functional as tf
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import math
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from transformer_yolov5 import BBTransformer, BoxEmbedder
import PIL.ImageOps  
import matplotlib.dates as mdates
#%%
# video_path = r"C:\Users\jankos\tyokansio\projektit\pierce-project\suturing_videos\videos\P13_SeeTrueProject_12_2_2020_18_18_54_sutures.mp4"
# video_path = r"C:\Users\jankos\tyokansio\projektit\pierce-project\forced_accuracy\Patient__2020-10-22-10-34-28-944-000.mpg"
# det_path = r"C:\Users\jankos\tyokansio\projektit\pierce-project\forced_accuracy\labels"

# video_path = r"C:\Users\jankos\tyokansio\projektit\pierce-project\edmonton_clips\testing\sample_17.mp4"
# det_path = r"C:\Users\jankos\tyokansio\projektit\pierce-project\edmonton_clips\testing\labels"

video_path = r"C:\Users\jankos\tyokansio\projektit\pierce-project\edmonton_clips\sample_9.mp4"
det_path = r"C:\Users\jankos\tyokansio\projektit\pierce-project\edmonton_clips\labels"

# video_path = r"C:\Users\jankos\tyokansio\projektit\pierce-project\suturing_videos\videos\P2_SeeTrueProject_12_2_2020_9_24_50_sutures.mp4"
# det_path = r"C:\Users\jankos\tyokansio\projektit\pierce-project\suturing_videos\detections\labels"
#%%
detections = glob.glob(det_path + '/sample_9*.txt')
# detections = glob.glob(det_path + '/*.txt')
fn = [int(re.findall('_(\d+).txt', x)[0]) for x in detections]
det_df = pd.DataFrame({'detections':detections, 'frame_n':fn})
det_df = det_df.sort_values(by='frame_n').reset_index(drop=True)
#%%
model = BBTransformer(img_size=100,d_embed=96, n_head=8,
                        n_layers=8, n_classes=2)
model.load_state_dict(torch.load("pierce_full_weights/w_last.pt"))
model.eval()
#%%
data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
#%%
videoname='edmonton-sample_9-test-clip'
vidread = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')#VideoWriter_fourcc('m', 'p', '4', 'v')
video_writer = cv2.VideoWriter(f'{videoname}_predictions.mp4', fourcc, 20, (640, 480))
start_frame = 1
vidread.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
i=start_frame
ret = True
stacks = []
participant = os.path.basename(det_path).split('_')[0]
#action_map = {0: 'pierce', 1: 'transport', 2: 'nothing'}
action_map = {0: 'not pierce', 1: 'pierce'}
last_detection = 1
max_gap = 20

needle_id = 3
width = 640
height = 480
actions = []
frames = []
while ret:
   pred_text = ""
   # print("heipparallaa")
   x = None
   pred = torch.Tensor([-1])
   ret, frame = vidread.read()
   if not ret:
       break
   frame = cv2.resize(frame, (width, height))
   # print(frame.shape)
   cf = det_df[det_df['frame_n'] == i]
   
   i+=1
   print("frame ", i)
   draw = True
   if (i-last_detection)>max_gap:
       # print("too long gap, clear stack")
       stacks = []
       pred_text = ""

   if len(cf)==0:
       draw = False
   else:
       detects = np.loadtxt(cf['detections'].item(), ndmin=2)
       detects = detects[detects[:,0] == needle_id]
   if draw and len(detects)>0:
       last_detection = i
       d = detects[:1,:].squeeze()
       # print(d.shape)
       x, w = int(d[1]*640), int(d[3]*640)
       y, h = int(d[2]*480), int(d[4]*480)
       box = frame[y-50:y+50, x-50:x+50]
       frame = cv2.rectangle(frame, (x-50,y-50), (x+50,y+50), (0,255,0))
       if box.shape == (100,100,3):
           stacks.append(box)
   if len(stacks)>11:
       st = np.vstack(stacks[-12:])
       st_show = st.copy()
       st = np.reshape(st, (12,-1,100,3))
       st_minishow = st[0,:,:,:].copy()
       st = st.transpose(0,3,1,2)
       # st = st[:, :, :, :]
       # print(st.shape)
       st = torch.from_numpy(st).float()
       st_minishow = st[0,:,:,:]
       bxs = []
       for k in range(1):
           b = tf.to_pil_image(st[k,:,:,:])
           b = PIL.ImageOps.invert(b)
           # print("b ", b.size)
           bx = data_transforms(b)
           bxs.append(bx)
       st = torch.stack(bxs)
       st = st.unsqueeze(dim=0)
       out = model(st)
       _, pred = torch.max(out, 1)
       # print("predicted ", pred.item(), out)
       pred_text = action_map[pred.item()]
       actions.append(pred_text)
       frames.append(i)
   frame = cv2.putText(frame, f"predicted action: {pred_text}", (150,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, 1)
   if x != None:
        frame = cv2.rectangle(frame, (x-50,y-50), (x+50,y+50), (0,255,0))
   print(frame.shape)
   video_writer.write(frame)
   # if i>6000:
   #     break
detections_df = pd.DataFrame({'frame':frames, 'action':actions})
detections_df.to_csv(f'{videoname}_detections.csv', index=None)
video_writer.release()
cv2.destroyAllWindows()
#%%
detections_df = pd.read_csv(f'{videoname}_detections.csv')
new_index = np.arange(1,detections_df['frame'].max()+1)
detections_df = detections_df.set_index('frame')
detections_df = detections_df.reindex(new_index)
detections_df = detections_df.reset_index()
#%%
detections_df['binary'] = (detections_df['action'] == 'pierce').astype(int)
detections_df['time'] = pd.to_datetime((detections_df['frame']/25), unit='s')
#%%
fig, ax = plt.subplots(figsize=(15,4))
ax.plot(detections_df['time'], detections_df['binary'])
ax.fill_between(detections_df['time'].values, detections_df['binary'].values, alpha=0.5)
ax.set_yticks([0,1])
ax.set_yticklabels(['not pierce', 'pierce'])
ax.set(xlabel='time [hh:mm:ss]')
ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30)) 
plt.xticks(rotation=90)