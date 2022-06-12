from __future__ import print_function
import glob
from itertools import chain
import os
import random
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from vit_pytorch.efficient import ViT
from torchsummary import summary

# Training settings
batch_size = 1
epochs = 10
lr = 3e-5
gamma = 0.7
seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
train_dir = '../simple_dataset/trajectory_mask/train'
test_dir = '../simple_dataset/trajectory_mask/test'

train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

labels = [path.split('/')[-1].split('.')[1] for path in train_list]

# Split
train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, random_state=seed)

# Image Augumentation
train_transforms = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),])

val_transforms = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),])

test_transforms = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224),transforms.ToTensor(),])

# Load Datesets
class Tra_mask_Dataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[1]
        label = 1 if label == "Y" else 0

        return img_transformed, label

train_data = Tra_mask_Dataset(train_list, transform=train_transforms)
valid_data = Tra_mask_Dataset(valid_list, transform=test_transforms)
test_data = Tra_mask_Dataset(test_list, transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

### Effecient Attention ###
# Linformer
efficient_transformer = Linformer(
    dim=128,
    seq_len=197,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

# Visual Transformer
model = ViT(dim=128, image_size=224, patch_size=16, num_classes=2, transformer=efficient_transformer, channels=1).to(device)
summary(model, (1, 224, 224))
#Training
criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.Adam(model.parameters(), lr=lr) # optimizer
scheduler = StepLR(optimizer, step_size=1, gamma=gamma) # scheduler

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - "
          f"val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")