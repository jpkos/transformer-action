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
file_folder = 'local_files/pierce_full/val'
files = glob.glob(os.path.join(file_folder, '*.jpeg'))
labels = []
for file in files:
    basename = os.path.basename(file)
    if basename.startswith('pierc'):
        labels.append('pierce')
    else:
        labels.append('not-pierce')
        
#%%
df = pd.DataFrame({'file':files, 'label':labels})
#%%
df.to_csv(os.path.join(file_folder, 'labels.csv'), index=None)