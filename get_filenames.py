#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 21:37:21 2021

@author: weiyunjiang
"""
import os
import pickle 

# train_data_dir = "./anime_data/train"
# train_filenames = os.listdir(train_data_dir) 
# with open('train_filenames', 'wb') as f:
#     pickle.dump(train_filenames, f)


# val_data_dir = "./anime_data/val"
# val_filenames = os.listdir(val_data_dir) 
# with open('val_filenames', 'wb') as f:
#     pickle.dump(val_filenames, f)
    
    
test_filenames = "1841032.png"
with open('test_filenames', 'wb') as f:
    pickle.dump(test_filenames, f)
    
pass