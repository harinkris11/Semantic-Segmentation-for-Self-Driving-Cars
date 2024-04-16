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


class DeepLabV3_vit(nn.Module):
    def __init__(self, num_classes=23):
        super(DeepLabV3_vit, self).__init__()

        encoder = models.vision_transformer.vit_b_16(pretrained=True)
        self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        
        # ASPP module
        self.aspp = ASPPModule(768, 256, rates=[6, 12, 18])
        
        # Decoder
        self.decoder = Decoder(256, 256, num_classes)
        
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        x = self.decoder(x)
        
        return x


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPPModule, self).__init__()
        
        # 1x1 convolution
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Dilated convolutions
        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2])
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 1x1 convolution to adjust the channel dimension
        self.conv_1x1_adjust = nn.Conv2d(1280, out_channels, kernel_size=1)
        
    def forward(self, x):
        # 1x1 convolution
        x_1x1 = self.conv_1x1(x)
        
        # Dilated convolutions
        x_3x3_1 = self.conv_3x3_1(x)
        x_3x3_2 = self.conv_3x3_2(x)
        x_3x3_3 = self.conv_3x3_3(x)
        
        # Global average pooling
        x_global_avg_pool = self.global_avg_pool(x)
        x_global_avg_pool = x_global_avg_pool.expand(-1, -1, x.size(2), x.size(3))
        x_global_avg_pool = self.conv_1x1(x_global_avg_pool)

        # Concatenate all paths
        out = torch.cat((x_1x1, x_3x3_1, x_3x3_2, x_3x3_3, x_global_avg_pool), dim=1)
        
        # Adjust the channel dimension
        out = self.conv_1x1_adjust(out)
        
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(Decoder, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(out_channels, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = F.interpolate(x, size = (160,240), mode='bilinear', align_corners=False)
        x = self.conv_2(x)
        x = self.conv_3(x)
        
        return x

# deeplab_vit_model = DeepLabV3_vit(num_classes=23).to(DEVICE)


# LEARNING_RATE = 1e-4
# num_epochs = 8
# train_batch,test_batch = get_images(data_dir, transform = t1, batch_size = 32)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(deeplab_vit_model.parameters(), lr=LEARNING_RATE)
# scaler = torch.cuda.amp.GradScaler()
# loss = 0
# train_loss_list = []
# test_loss_list = []
# test_interval = 2
# # Training loop

# for epoch in range(num_epochs):
#     loop = tqdm(enumerate(train_batch), total=len(train_batch))
#     deeplab_vit_model.train()
#     train_loss = 0
#     for batch_idx, (data, targets) in loop:
#         data = data.to(DEVICE)
#         targets = targets.to(DEVICE)
#         targets = targets.type(torch.long)

#         # Forward pass
#         with torch.cuda.amp.autocast():
#             predictions = deeplab_vit_model(data)
#             loss = loss_fn(predictions, targets)
#         #deeplab_vit_loss_list.append(loss)
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         loop.set_postfix(loss=loss.item())
#         train_loss += loss.detach().item()
        
#     train_loss = train_loss/len(train_batch)
#     train_loss_list.append(train_loss)
    
    
# #plot_loss(deeplab_vit_loss_list)
#     if epoch % test_interval == 0:
#         loop_test = tqdm(enumerate(test_batch), total=len(test_batch))
#         deeplab_vit_model.eval()
#         test_loss = 0
#         for batch_idx, (data, targets) in loop_test:
#             data = data.to(DEVICE)
#             targets = targets.to(DEVICE)
#             targets = targets.type(torch.long)
#             predictions = deeplab_vit_model(data)
#             loss = loss_fn(predictions, targets)
#             loop_test.set_postfix(loss=loss.item())
#             test_loss += loss.detach().item()     

#         test_loss = test_loss/len(test_batch)
#         test_loss_list.append(test_loss)

    
