from torch.cuda.amp import autocast, GradScaler
import torch

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

file_names = train_loader.get('file_names')

train_dogs_path = file_names.get('train_dogs_path')
train_cats_path = file_names.get('train_cats_path')

train_ds_catsdogs = DataSet2Class(train_dogs_path, train_cats_path)

train_loader = torch.utils.data.DataLoader(
    train_ds_catsdogs, shuffle=train_loader.get('shuffle'),
    batch_size=train_loader.get('batch_size'), num_workers=train_loader.get('num_workers'),
    drop_last=train_loader.get('drop_last')
)

CNNet = ConvNet()

loss_fn = get_losser(network_options.get('loss'))
optimizer = get_optimizer(CNNet.parameters(), network_options.get('optimizer'))

device = network_options.get('device')
use_amp = network_options.get('use_amp')
scaler = torch.cuda.amp.GradScaler()
epochs = network_options.get('epochs')

CNNet = CNNet.to(device)
loss_fn = loss_fn.to(device)

start_time = time.time()

for epoch in range(epochs):
    loss_val = 0
    acc_val = 0
    for sample in train_loader:   # (pbar := tqdm(train_loader))
        img, lbl = sample['img'], sample['label']
        lbl = F.one_hot(lbl, 2).float()
        img = img.to(device)
        lbl = lbl.to(device)
        optimizer.zero_grad()

        with autocast(use_amp):
            pred = CNNet(img)
            loss = loss_fn(pred, lbl)
        scaler.scale(loss).backward()
        loss_item = loss.item()
        loss_val += loss_item

        scaler.step(optimizer)
        scaler.update()

        acc_current = accuracy(pred.cpu().float(), lbl.cpu().float())
        acc_val += acc_current

    print(f"Epoch : {epoch+1}")
    print(f"Loss : {loss_val / len(train_loader)}")
    print(f"Acc : {acc_val / len(train_loader)}")

print(f'Full time learning : {time.time() - start_time}')

curDate = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
print(curDate)
path_name = "" + curDate + "_" + device.upper() + ".pth"
torch.save(CNNet.state_dict(), path_name)
