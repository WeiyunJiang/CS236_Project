#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 21:53:21 2021

@author: weiyunjiang
"""
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Compose, ToTensor
from PIL import Image
import matplotlib.pyplot as plt

import pickle 

class AnimeDataset(Dataset):
    def __init__(self, split='train', resolution=(64, 64), downsample=True):
        # 512 x 512
        self.downsample = downsample
        self.resolution = resolution
        
        if split == "train":
            self.root_dir = "./anime_data/train/"
            with open('train_filenames', 'rb') as f:
                self.filenames = pickle.load(f)
            self.transform_image = Compose([
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        elif split == "test":
            self.root_dir = "./anime_data/val/"
            with open('val_filenames', 'rb') as f:
                self.filenames = pickle.load(f)
        else:
            raise NotImplementedError('Not implemented for name={split}')
            
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        pth = self.root_dir + filename
        image_pair = Image.open(pth)
        width, height = image_pair.size
        colored = image_pair.crop((0, 0, width//2, height))
        sketch = image_pair.crop((width//2, 0, width, height))
        if self.downsample: # downsample the image to reduce storage
            colored = colored.resize(self.resolution)
            sketch = sketch.resize(self.resolution)
        colored = self.transform_image(colored.copy())
        sketch = self.transform_image(sketch.copy())
        sample = {'sketch': sketch, 'colored': colored}
        return sample
if __name__ == "__main__":
    animedataset = AnimeDataset()
    sample  = animedataset[2]
    plt.figure()
    plt.imshow(sample['sketch'].permute(1,2,0))
    plt.figure()
    plt.imshow(sample['colored'].permute(1,2,0))
    pass
