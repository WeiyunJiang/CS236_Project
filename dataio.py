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
import torchvision.transforms.functional as TF
import random
import pickle 

class AnimeDataset(Dataset):
    def __init__(self, split='train', resolution=(64, 64), data_aug=True, downsample=True):
        # 512 x 512
        self.downsample = downsample
        self.resolution = resolution
        self.data_aug = data_aug
        self.split = split
        if split == "train":
            self.root_dir = "./anime_data/train/"
            with open('train_filenames', 'rb') as f:
                self.filenames = pickle.load(f)
        elif split == "test":
            self.root_dir = "./anime_data/test/"
            with open('test_filenames', 'rb') as f:
                self.filenames = pickle.load(f)
        else:
            raise NotImplementedError('Not implemented for name={split}')
        if data_aug is True:
            print("Using Data Augmentation!")
        else:
            print("No Data Augmentation!")
        self.transform_image = Compose([
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])    
            
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # filename = self.filenames[idx]
        filename = "1841032.png"
        pth = self.root_dir + filename
        image_pair = Image.open(pth)
        width, height = image_pair.size
        colored = image_pair.crop((0, 0, width//2, height))
        sketch = image_pair.crop((width//2, 0, width, height))
        if self.downsample: # downsample the image to reduce storage
            colored = colored.resize(self.resolution)
            sketch = sketch.resize(self.resolution)
        if self.data_aug is True and self.split == "train":
            # hflip with p = 0.5
            if random.random() > 0.5:
                colored = TF.hflip(colored)
                sketch = TF.hflip(sketch)
                
                
            # (rotate -10 to 10 degrees with p=0.8)
            if random.random() > 0.2:
                angle_bound = 10
                random_angle = (random.random() - 0.5) * 2 * angle_bound
                colored = colored.rotate(random_angle)
                sketch = sketch.rotate(random_angle)
                
        colored = self.transform_image(colored.copy())
        sketch = self.transform_image(sketch.copy())
        sample = {'sketch': sketch, 'colored': colored}
        return sample
    
if __name__ == "__main__":

    
    train_dataset = AnimeDataset(split='train', resolution = (256, 256), data_aug=True)
    test_dataset = AnimeDataset(split='test', resolution = (256, 256), data_aug=True)
    
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    for i, batch in enumerate(train_data_loader):
        print(i)
        print(batch['sketch'].shape) # 64, 3, 64, 64
        plt.figure()
        plt.imshow(batch['sketch'][0].permute(1,2,0))
        plt.figure()
        plt.imshow(batch['colored'][0].permute(1,2,0))
    pass
