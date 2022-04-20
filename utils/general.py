# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:39:05 2022

@author: jankos
"""

import glob
import os
from pathlib import Path
import re
#%%

def find_last_run(runs):
    vals = [re.search(r'(\d+)', rf'{x}') for x in runs]
    vals = [0 if x == None else int(x.group(0)) for x in vals]
    return max(vals)

def create_save_folder(path):
    path = Path(path)
    parent = path.parents[0]
    if path.exists():
        runs = glob.glob(str(parent) + f'/{path.name}*')
        last_run = find_last_run(runs)
        save_path = Path(f'{path}{last_run+1}')
    else:
        save_path = Path(f'{path}')
    save_path.mkdir(parents=True, exist_ok=True)
    
    return save_path
    

