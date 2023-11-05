import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv
import cv2

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

from DataSet import DataSet2Class
from NN import ConvNet
from Functions import *


train_dogs_path = 'C:/Users/relax_fm/PycharmProjects/CNN/dataset/training_set/dogs'
train_cats_path = 'C:/Users/relax_fm/PycharmProjects/CNN/dataset/training_set/cats'
test_dogs_path = './dataset/test_set/dogs'
test_cats_path = './dataset/test_set/cats'

train_ds_catsdogs = DataSet2Class(train_dogs_path, train_cats_path)
test_ds_catsdogs = DataSet2Class(test_dogs_path, test_cats_path)

# plt.figure(figsize=(8, 8))
# plt.imshow(train_ds_catsdogs[0]['img'].numpy().transpose((1, 2, 0)))
# plt.show()


batch_size = 16

train_loader = torch.utils.data.DataLoader(
    train_ds_catsdogs, shuffle=True,
    batch_size=batch_size, num_workers=0, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    train_ds_catsdogs, shuffle=True,
    batch_size=batch_size, num_workers=0, drop_last=False
)

CNNet = ConvNet()

# print(count_parametrs(CNNet))

# for sample in train_loader:
#     img = sample['img']
#     label = sample['label']
#     print(CNNet(img))
#     break

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNet.parameters(), lr=0.0001, betas=(0.9, 0.999))

epochs = 20

for epoch in range(epochs):
    loss_val = 0
    acc_val = 0
    for sample in train_loader:   # (pbar := tqdm(train_loader))
        img, lbl = sample['img'], sample['label']
        optimizer.zero_grad()

        lbl = F.one_hot(lbl, 2).float()
        pred = CNNet(img)

        # print(pred)
        # print(lbl)
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

curDate = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
print(curDate)
path_name = ""+curDate+".pth"
torch.save(CNNet.state_dict(), path_name)
