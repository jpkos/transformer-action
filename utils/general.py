# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:39:05 2022

@author: jankos
"""

import glob
import os
from pathlib import Path
import re
import seaborn as sns
import matplotlib.pyplot as plt
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
    
def print_model_params(model):
    ps = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    print(f'Model parameters: {ps}')

def plot_results(df):
    fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(10,10))
    ax = ax.flatten()
    for i, col in enumerate(['accuracy', 'precision', 'recall', 'f1']):
        sns.lineplot(x='epoch', y=col, hue='mode', data=df, ax=ax[i], legend=(i==0))
    return fig, ax