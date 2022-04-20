# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 17:37:17 2021

@author: jankos
"""

import cv2
import numpy as np
import pandas as pd
from video_clipper import VideoFrameDrill, VideoClipReader, VideoClipApprover, VideoClipCreator
#%%ENT
cols = ['annotation_type', 'time_min1', 'time_sec1', 'time_ms1',
                            'time_min2', 'time_sec2', 'time_ms2',
                            'time_min3', 'time_sec3', 'time_ms3', 'action']
anno_df = pd.read_csv('P8_ENT/770603.txt', header=None,
                      sep=r'\t+', names=cols, engine='python')
anno_df['frame'] = (anno_df['time_sec1']*30).astype(int)
#%%
vidread = cv2.VideoCapture('P8_ENT/P8_gazeVideo_0.mp4')
ret, frame = vidread.read()

# grasps = pd.read_csv('train_videos/P6_grab_events_LHRH_2020_5_5_annotated_events.csv')
# grasps = grasps.drop(index=0).reset_index(drop=True)
i = 0
dets = pd.read_csv('P8_ENT/P8_detections.txt',
                   names=['frame', 'class', 'x', 'y','w', 'h',
                          'conf'],
                   sep=' ')
dets['frame'] = (dets['frame']).astype(int)
#%%
dets['drill'] = pd.cut(dets['frame'], bins=anno_df['frame'],
                          labels=anno_df['action'].iloc[:-1], ordered=False)
# grasps['frame'] = grasps['frame'].astype(int)
#%%
def clip_condition(clip):
    return all(map(lambda x: x == clip[0], clip))

drill_classes = {0:'drill', 1:'suction', 2:'hook'}

#%%
dets['class'] = dets['class'].map(drill_classes)
dets = dets[dets['class'] == 'drill'].reset_index()
dets[['x', 'w']] = dets[['x', 'w']]*640
dets[['y', 'h']] = dets[['y', 'h']]*480
#%%

#%%
vidread = cv2.VideoCapture('P8_ENT/P8_gazeVideo_0.mp4')
ret, frame = vidread.read()
px_pad = 50
clip_reader = VideoClipReader(clip_length=12)
VCA = VideoClipApprover([clip_reader], condition=clip_condition)
VCC = VideoClipCreator(approver=VCA)

fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
vidwrite = cv2.VideoWriter('test.mp4', fourcc, 15, (px_pad*2,px_pad*2))
# vidwrite = cv2.VideoWriter('test.mp4', fourcc, 15, (640,480))
fn = 1
p_lim = 0.2
while ret:
    # vidwrite.write(frame)
    dets_fn = dets[dets['frame'] == fn]
    for i, det in dets_fn.iterrows():
        y0 = int(det['y']) - px_pad
        y1 = int(det['y']) + px_pad
        x0 = int(det['x']) - px_pad
        x1 = int(det['x']) + px_pad
        box = frame[y0:y1,x0:x1,...]
        clip_reader.append(VideoFrameDrill(det['drill'], box))
        if isinstance(VCC.clip, np.ndarray) and fn%12==0:
            stacked = VCC.clip
            p = np.random.rand(1)
            dest = 'val' if p<p_lim else 'train'
            cv2.imwrite('P8_ENT/clips/{}/{}_{}.jpeg'.format(dest, det['drill'], fn), stacked)
    # cv2.circle(frame, (int(det['x']), int(det['y'])), 5, (0,255,0))
    vidwrite.write(box)
    ret, frame = vidread.read()
    fn += 1
    if fn > 15000:
        break
    
vidwrite.release()
