import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn.functional as F
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self,img_dir,transform = None):
        self.transforms = transform
        image_paths = [i+'/CameraRGB' for i in img_dir]
        seg_paths = [i+'/CameraSeg' for i in img_dir]
        self.images,self.masks = [],[]
        for i in image_paths:
            imgs = os.listdir(i)
            self.images.extend([i+'/'+img for img in imgs])
        for i in seg_paths:
            masks = os.listdir(i)
            self.masks.extend([i+'/'+mask for mask in masks])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        img = np.array(Image.open(self.images[index]))
        mask = np.array(Image.open(self.masks[index]))
        if self.transforms is not None:
            aug = self.transforms(image=img,mask=mask)
            img = aug['image']
            mask = aug['mask']
            mask = torch.max(mask,dim=2)[0]
        return img,mask



def get_images(
        image_dir,
        transform = None,
        batch_size=1,
        shuffle=True,
        pin_memory=True, 
        train_size = 0.8
    ):

    data = SegmentationDataset(image_dir, transform = t1)
    train_size = int(train_size * data.__len__())
    test_size = data.__len__() - train_size
    train_dataset, test_dataset = random_split(data, [train_size, test_size])
    train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    test_batch = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    return train_batch,test_batch