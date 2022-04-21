# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:20:11 2022

@author: jankos
"""

import pandas as pd
import numpy as np
import re
import glob
import os
#%%
file_folder = 'local_files/pierce_full/train'
files = glob.glob(os.path.join(file_folder, '*.jpeg'))
labels = []
for file in files:
    basename = os.path.basename(file)
    if basename.startswith('pierc'):
        labels.append('pierce')
    else:
        labels.append('not-pierce')
#%%
file_folder = 'local_files/action_dataset/val'
files = glob.glob(os.path.join(file_folder, '*.jpeg'))
labels = []
for file in files:
    basename = os.path.basename(file)
    labels.append(basename.split('_')[1])
#%%
file_folder = 'local_files/toy_dataset/test'
files = glob.glob(os.path.join(file_folder, '*.jpeg'))
labels = []
for file in files:
    basename = os.path.basename(file)
    labels.append(basename.split('_')[0])
        
#%%
df = pd.DataFrame({'file':files, 'label':labels})
#%%
df.to_csv(os.path.join(file_folder, 'labels.csv'), index=None)