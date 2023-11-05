import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv
import cv2

import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(3, 32, 3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=0)

        self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 50)
        self.linear2 = nn.Linear(50, 2)

    def forward(self, x):
        #print(x.shape)
        out = self.conv0(x)
        out = self.act(out)
        out = self.maxpool(out)
        #print(out.shape)
        out = self.conv1(out)
        out = self.act(out)
        out = self.maxpool(out)
        #print(out.shape)
        out = self.conv2(out)
        out = self.act(out)
        out = self.maxpool(out)
        #print(out.shape)
        out = self.conv3(out)
        out = self.act(out)
        #print(out.shape)

        out = self.adaptivepool(out)
        #print(out.shape)
        out = self.flatten(out)
        #print(out.shape)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        #print(out.shape)
        # print(out)

        return out
