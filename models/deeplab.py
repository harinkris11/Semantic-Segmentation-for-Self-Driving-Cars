import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()

        self.aspp_layers = nn.ModuleList()
        for rate in atrous_rates:
            aspp_layer = nn.Conv2d(in_channels, 1, kernel_size=3, padding=rate, dilation=rate)
            self.aspp_layers.append(aspp_layer)

        self.img_level = nn.AdaptiveAvgPool2d(1)
        self.img_level_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.upsample = nn.Upsample(size=(10, 15), mode='bilinear', align_corners=False)

    def forward(self, x):
        aspp_outputs = []
        for layer in self.aspp_layers:
            aspp_output = nn.ReLU()(layer(x))
            aspp_outputs.append(aspp_output)

        img_level = self.img_level(x)
        img_level = self.img_level_conv(img_level)
        img_level = self.upsample(img_level)

        aspp_outputs.append(img_level)

        concat = torch.cat(aspp_outputs, dim=1)
        return concat
    
    
class DeepLabModel(nn.Module):
    def __init__(self, in_ch, out_ch, atrous_rates=[6, 12, 18, 24]):
        super(DeepLabModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.aspp = ASPP(512, atrous_rates)

        self.upsample = nn.ConvTranspose2d(5, 23, kernel_size=16, stride=16)
        self.conv_fin = nn.Conv2d(23, 23, kernel_size=1)

        self.relu = nn.ReLU()
        

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv1 = self.relu(self.conv2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.relu(self.conv3(pool1))
        conv2 = self.relu(self.conv4(conv2))
        pool2 = self.pool2(conv2)

        conv3 = self.relu(self.conv5(pool2))
        conv3 = self.relu(self.conv6(conv3))
        pool3 = self.pool3(conv3)

        conv4 = self.relu(self.conv7(pool3))
        conv4 = self.relu(self.conv8(conv4))
        pool4 = self.pool4(conv4)

        conv5 = self.relu(self.conv9(pool4))
        conv5 = self.relu(self.conv10(conv5))

        aspp = self.aspp(conv5)

        upsample = self.upsample(aspp)
        conv_fin = self.conv_fin(upsample)

        return conv_fin