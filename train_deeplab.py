import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torchsummary import summary
import torch.nn.functional as F
import torchvision.models as models
from dataset.segmentationdataset import SegmentationDataset
from dataset.segmentationdataset import get_images
from models.deeplab import DeepLabModel


def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)),axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


if __name__ == "__main__":


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    deeplab_model = DeepLabModel(3,23).to(DEVICE)

    LEARNING_RATE = 1e-4
    num_epochs = 50
    batch_size = 32
    train_batch,test_batch = get_images(data_dir, transform = t1, batch_size = batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(deeplab_model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    deeplab_loss_list = []
    # Training loop
    loss = 0
    train_loss_list = []
    test_loss_list = []
    test_interval = 10
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_batch), total=len(train_batch))
        
        deeplab_model.train()
        train_loss = 0
        for batch_idx, (data, targets) in loop:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            targets = targets.type(torch.long)
            with torch.cuda.amp.autocast():
                predictions = deeplab_model(data)
                loss = loss_fn(predictions, targets)
            optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.detach().item()
            loop.set_postfix(loss=loss.item())
        train_loss = train_loss/len(train_batch)
        train_loss_list.append(train_loss)
        
        
        if epoch % test_interval == 0:
            loop_test = tqdm(enumerate(test_batch), total=len(test_batch))
            deeplab_model.eval()
            test_loss = 0
            for batch_idx, (data, targets) in loop_test:
                data = data.to(DEVICE)
                targets = targets.to(DEVICE)
                targets = targets.type(torch.long)
                predictions = deeplab_model(data)
                loss = loss_fn(predictions, targets)
                test_loss += loss.detach().item()     
                loop_test.set_postfix(loss=loss.item())
            
            test_loss = test_loss/len(test_batch)
            test_loss_list.append(test_loss)
                
    check_accuracy(train_batch, deeplab_model)  
    check_accuracy(test_batch, deeplab_model)              
