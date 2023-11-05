import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv
import cv2

import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm

def count_parametrs(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()