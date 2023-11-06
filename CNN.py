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
train_loader = dataset_options.get('train_loader')
test_loader = dataset_options.get('test_loader')
file_names = dataset_options.get('file_names')

train_dogs_path = file_names.get('train_dogs_path')
train_cats_path = file_names.get('train_cats_path')
test_dogs_path = file_names.get('test_dogs_path')
test_cats_path = file_names.get('test_cats_path')

train_ds_catsdogs = DataSet2Class(train_dogs_path, train_cats_path)
test_ds_catsdogs = DataSet2Class(test_dogs_path, test_cats_path)

# plt.figure(figsize=(8, 8))
# plt.imshow(train_ds_catsdogs[0]['img'].numpy().transpose((1, 2, 0)))
# plt.show()


batch_size = 16

train_loader = torch.utils.data.DataLoader(
    train_ds_catsdogs, shuffle=train_loader.get('shuffle'),
    batch_size=train_loader.get('batch_size'), num_workers=train_loader.get('num_workers'),
    drop_last=train_loader.get('drop_last')
)
test_loader = torch.utils.data.DataLoader(
    test_ds_catsdogs, shuffle=test_loader.get('shuffle'),
    batch_size=test_loader.get('batch_size'), num_workers=test_loader.get('num_workers'),
    drop_last=test_loader.get('drop_last')
)

CNNet = ConvNet()

# print(count_parametrs(CNNet))

# for sample in train_loader:
#     img = sample['img']
#     label = sample['label']
#     print(CNNet(img))
#     break

loss_fn = get_losser(network_options.get('loss'))
optimizer = get_optimizer(CNNet.parameters(), network_options.get('optimizer'))

device = network_options.get('device')
use_amp = network_options.get('use_amp')
epochs = network_options.get('epochs')

start_time = time.time()

for epoch in range(epochs):
    loss_val = 0
    acc_val = 0
    for sample in train_loader:   # (pbar := tqdm(train_loader))
        img, lbl = sample['img'], sample['label']
        optimizer.zero_grad()

        lbl = F.one_hot(lbl, 2).float()

        with autocast(use_amp, dtype=torch.float16):
            pred = CNNet(img)
            loss = loss_fn(pred, lbl)

        loss.backward()
        loss_item = loss.item()
        loss_val += loss_item

        optimizer.step()

        acc_current = accuracy(pred, lbl)
        acc_val += acc_current

    # pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
    print(f"Epoch : {epoch+1}")
    print(f"Loss : {loss_val / len(train_loader)}")
    print(f"Acc : {acc_val / len(train_loader)}")

print(f'Full time learning : {time.time() - start_time}')

curDate = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
print(curDate)
path_name = "" + curDate + "_" + device.upper() + ".pth"
torch.save(CNNet.state_dict(), path_name)
