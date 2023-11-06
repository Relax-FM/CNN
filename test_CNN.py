import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torchvision as tv
import cv2

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import yaml
import datetime

from DataSet import DataSet2Class
from NN import ConvNet
from Functions import *
from Losses import get_losser
from Optimizer import get_optimizer

options_path = 'config.yml'
with open(options_path, 'r') as options_stream:
    options = yaml.safe_load(options_stream)

network_options = options.get('network')
dataset_options = options.get('dataset')

test_loader = dataset_options.get('test_loader')
file_names = test_loader.get('file_names')

test_dogs_path = file_names.get('test_dogs_path')
test_cats_path = file_names.get('test_cats_path')

test_ds_catsdogs = DataSet2Class(test_dogs_path, test_cats_path)

test_loader = torch.utils.data.DataLoader(
    test_ds_catsdogs, shuffle=test_loader.get('shuffle'),
    batch_size=test_loader.get('batch_size'), num_workers=test_loader.get('num_workers'),
    drop_last=test_loader.get('drop_last')
)

CNNet = ConvNet()

path_name = network_options.get('test_model_path')

print(f'Start with model {path_name}')

CNNet.load_state_dict(torch.load(path_name))

loss_val = 0
acc_val = 0
loss_fn = get_losser(network_options.get('loss'))
for sample in test_loader:
    with torch.no_grad():
        img, label = sample['img'], sample['label']

        label = F.one_hot(label, 2).float()
        pred = CNNet(img)

        loss = loss_fn(pred, label)
        loss_item = loss.item()
        loss_val += loss_item

        acc_current = accuracy(pred, label)
        acc_val += acc_current

    print(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
print(f'Loss of all DS: {loss_val/len(test_loader)}')
print(f'Acc of all DS: {acc_val/len(test_loader)}')