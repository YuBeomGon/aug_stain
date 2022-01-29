import torch 
import torchvision 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 
import matplotlib.image as image 
import numpy as np

import os
import pandas as pd
import albumentations as A
import albumentations.pytorch
import cv2
import math

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from augment import HEColorAugment

IMAGE_SIZE = 256
train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
    A.OneOf([
        A.HorizontalFlip(p=.8),
        A.VerticalFlip(p=.8),
        A.RandomRotate90(p=.8)]
    ),
    A.RandomResizedCrop(height=IMAGE_SIZE,width=IMAGE_SIZE,scale=[0.95,1.05],ratio=[0.95,1.05],p=0.5),
#     A.OneOf([
#         A.transforms.JpegCompression(quality_lower=99, quality_upper=100, p=.5),
#         A.transforms.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.2, alpha_coef=0.08, p=.5),
#         A.transforms.RandomGamma(gamma_limit=(80, 120), eps=None, p=.5),
#     ]),
    A.transforms.ColorJitter(brightness=0.05, contrast=0.2, saturation=0.2, hue=0.02, p=.8),
#     HEColorAugment(sigma1=0.05, sigma2=3, theta=6., p=.8),
#     A.pytorch.ToTensor(),
], p=1.0) 


val_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
#     A.pytorch.ToTensor(), 
], p=1.0) 


class CDataset(Dataset):
    def __init__(self, df, defaultpath='/home/beomgon/Dataset/patches/', transform=None):
        self.df = df
        self.transform = transform
        self.dir = defaultpath

    def __len__(self):
        return len(self.df)   

    def __getitem__(self, idx):
        path = self.df.iloc[idx, 4]
#         print(pid)

        image = cv2.imread(self.dir + path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = (image.astype(np.float32)-128.)/128.
        
#         if image is uint8, normalization by 255 is done automatically by albumebtation(ToTensor method)
        if self.transform:
            timage = self.transform(image=image)
            image = timage['image']
        
#         image =  torch.tensor(image, dtype=torch.float32)
        image = (torch.tensor(image, dtype=torch.float32)-128)/128
        image = image.permute(2,0,1)
            
        label = self.df.iloc[idx, 5]
        return image, label, path




