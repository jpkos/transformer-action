# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:44:57 2022

@author: jankos
"""

import numpy as np
import glob
import os
from shutil import copyfile
import argparse
from pathlib import Path
#%%
def main(args):
    files = np.array(glob.glob(os.path.join(args.source_folder, '*.' + args.ext)))
    n_files = len(files)
    n_validation_files = int(args.val_ratio*n_files)
    mask = np.random.choice(n_files, n_validation_files, replace=False)
    print(mask)
    validation_files = files[mask]
    train_files = np.delete(files, mask)
    train_path = os.path.join(args.source_folder, 'train')
    val_path = os.path.join(args.source_folder, 'val')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)
    for file in validation_files:
        copyfile(file, os.path.join(val_path, os.path.basename(file)))
    for file in train_files:
        copyfile(file, os.path.join(train_path, os.path.basename(file)))
    print(f'train files: {len(train_files)}\nval files: {len(validation_files)}')
    
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_folder', type=str)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--ext', type=str, default='jpg')
    args = parser.parse_args()
    main(args)
    