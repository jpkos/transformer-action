# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:52:07 2022

@author: jankos
"""

import yaml
import pandas as pd
import numpy as np
#%%

with open('params/params.yaml', 'r') as y:
    d = yaml.safe_load(y)