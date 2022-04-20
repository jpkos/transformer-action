# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:10:13 2022

@author: jankos
"""

import numpy as np
import cv2
import os
import glob
#%%
N_CLIPS = 15
n_img = 12
img_size = 100
dr = np.round(img_size/n_img).astype(int)
dr_map = {'left':(-1*dr, 0), 'right':(dr, 0), 'up':(0,-1*dr), 'down':(0,dr)}
for i in range(N_CLIPS):
    clip = []
    direction = np.random.choice(['left', 'right', 'up', 'down'])
    v = dr_map[direction]
    if direction == 'up':
        pos_init = (np.random.randint(0, img_size), 100)
    elif direction == 'down':
        pos_init = (np.random.randint(0, img_size), 0)
    elif direction == 'left':
        pos_init = (img_size, np.random.randint(0, img_size))
    elif direction == 'right':
        pos_init = (0, np.random.randint(0, img_size))
    color = tuple(np.random.randint(0,255, size=(3,)).astype(int))
    color = ( int(color[0]),int (color[1]), int(color[2]))
    radius = np.random.randint(10,15)
    for j in range(1,n_img+1):
        new_pos = np.array(pos_init) + np.array(v)*j
        bg = np.ones((img_size,img_size,3))*255
        bg = cv2.circle(bg, (new_pos[0], new_pos[1]), radius=radius, color=color, thickness=-1)
        bg = bg.astype(np.uint8)
        clip.append(bg)
        # cv2.imshow('t', bg)
    print(direction)
    clip = np.vstack(clip)
    cv2.imwrite(f'local_files/toy_dataset/{direction}_{i}.jpeg', clip)
    
    