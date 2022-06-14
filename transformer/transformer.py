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


model = ViT(dim=128, image_size=224, patch_size=16, num_classes=2, transformer=efficient_transformer, channels=1).to(device)
# summary(model, (1, 224, 224), batch_size=1)

# Visual Transformer
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#          Rearrange-1              [1, 196, 256]               0
#             Linear-2              [1, 196, 128]          32,896
#          LayerNorm-3              [1, 197, 128]             256
#             Linear-4              [1, 197, 128]          16,384
#             Linear-5              [1, 197, 128]          16,384
#             Linear-6              [1, 197, 128]          16,384
#            Dropout-7            [1, 8, 197, 64]               0
#             Linear-8              [1, 197, 128]          16,512
# LinformerSelfAttention-9              [1, 197, 128]               0
#           PreNorm-10              [1, 197, 128]               0
#         LayerNorm-11              [1, 197, 128]             256
#            Linear-12              [1, 197, 512]          66,048
#              GELU-13              [1, 197, 512]               0
#           Dropout-14              [1, 197, 512]               0
#            Linear-15              [1, 197, 128]          65,664
#       FeedForward-16              [1, 197, 128]               0
#           PreNorm-17              [1, 197, 128]               0
#         LayerNorm-18              [1, 197, 128]             256
#            Linear-19              [1, 197, 128]          16,384
#            Linear-20              [1, 197, 128]          16,384
#            Linear-21              [1, 197, 128]          16,384
#           Dropout-22            [1, 8, 197, 64]               0
#            Linear-23              [1, 197, 128]          16,512
# LinformerSelfAttention-24              [1, 197, 128]               0
#           PreNorm-25              [1, 197, 128]               0
#         LayerNorm-26              [1, 197, 128]             256
#            Linear-27              [1, 197, 512]          66,048
#              GELU-28              [1, 197, 512]               0
#           Dropout-29              [1, 197, 512]               0
#            Linear-30              [1, 197, 128]          65,664
#       FeedForward-31              [1, 197, 128]               0
#           PreNorm-32              [1, 197, 128]               0
#         LayerNorm-33              [1, 197, 128]             256
#            Linear-34              [1, 197, 128]          16,384
#            Linear-35              [1, 197, 128]          16,384
#            Linear-36              [1, 197, 128]          16,384
#           Dropout-37            [1, 8, 197, 64]               0
#            Linear-38              [1, 197, 128]          16,512
# LinformerSelfAttention-39              [1, 197, 128]               0
#           PreNorm-40              [1, 197, 128]               0
#         LayerNorm-41              [1, 197, 128]             256
#            Linear-42              [1, 197, 512]          66,048
#              GELU-43              [1, 197, 512]               0
#           Dropout-44              [1, 197, 512]               0
#            Linear-45              [1, 197, 128]          65,664
#       FeedForward-46              [1, 197, 128]               0
#           PreNorm-47              [1, 197, 128]               0
#         LayerNorm-48              [1, 197, 128]             256
#            Linear-49              [1, 197, 128]          16,384
#            Linear-50              [1, 197, 128]          16,384
#            Linear-51              [1, 197, 128]          16,384
#           Dropout-52            [1, 8, 197, 64]               0
#            Linear-53              [1, 197, 128]          16,512
# LinformerSelfAttention-54              [1, 197, 128]               0
#           PreNorm-55              [1, 197, 128]               0
#         LayerNorm-56              [1, 197, 128]             256
#            Linear-57              [1, 197, 512]          66,048
#              GELU-58              [1, 197, 512]               0
#           Dropout-59              [1, 197, 512]               0
#            Linear-60              [1, 197, 128]          65,664
#       FeedForward-61              [1, 197, 128]               0
#           PreNorm-62              [1, 197, 128]               0
#         LayerNorm-63              [1, 197, 128]             256
#            Linear-64              [1, 197, 128]          16,384
#            Linear-65              [1, 197, 128]          16,384
#            Linear-66              [1, 197, 128]          16,384
#           Dropout-67            [1, 8, 197, 64]               0
#            Linear-68              [1, 197, 128]          16,512
# LinformerSelfAttention-69              [1, 197, 128]               0
#           PreNorm-70              [1, 197, 128]               0
#         LayerNorm-71              [1, 197, 128]             256
#            Linear-72              [1, 197, 512]          66,048
#              GELU-73              [1, 197, 512]               0
#           Dropout-74              [1, 197, 512]               0
#            Linear-75              [1, 197, 128]          65,664
#       FeedForward-76              [1, 197, 128]               0
#           PreNorm-77              [1, 197, 128]               0
#         LayerNorm-78              [1, 197, 128]             256
#            Linear-79              [1, 197, 128]          16,384
#            Linear-80              [1, 197, 128]          16,384
#            Linear-81              [1, 197, 128]          16,384
#           Dropout-82            [1, 8, 197, 64]               0
#            Linear-83              [1, 197, 128]          16,512
# LinformerSelfAttention-84              [1, 197, 128]               0
#           PreNorm-85              [1, 197, 128]               0
#         LayerNorm-86              [1, 197, 128]             256
#            Linear-87              [1, 197, 512]          66,048
#              GELU-88              [1, 197, 512]               0
#           Dropout-89              [1, 197, 512]               0
#            Linear-90              [1, 197, 128]          65,664
#       FeedForward-91              [1, 197, 128]               0
#           PreNorm-92              [1, 197, 128]               0
#         LayerNorm-93              [1, 197, 128]             256
#            Linear-94              [1, 197, 128]          16,384
#            Linear-95              [1, 197, 128]          16,384
#            Linear-96              [1, 197, 128]          16,384
#           Dropout-97            [1, 8, 197, 64]               0
#            Linear-98              [1, 197, 128]          16,512
# LinformerSelfAttention-99              [1, 197, 128]               0
#          PreNorm-100              [1, 197, 128]               0
#        LayerNorm-101              [1, 197, 128]             256
#           Linear-102              [1, 197, 512]          66,048
#             GELU-103              [1, 197, 512]               0
#          Dropout-104              [1, 197, 512]               0
#           Linear-105              [1, 197, 128]          65,664
#      FeedForward-106              [1, 197, 128]               0
#          PreNorm-107              [1, 197, 128]               0
#        LayerNorm-108              [1, 197, 128]             256
#           Linear-109              [1, 197, 128]          16,384
#           Linear-110              [1, 197, 128]          16,384
#           Linear-111              [1, 197, 128]          16,384
#          Dropout-112            [1, 8, 197, 64]               0
#           Linear-113              [1, 197, 128]          16,512
# LinformerSelfAttention-114              [1, 197, 128]               0
#          PreNorm-115              [1, 197, 128]               0
#        LayerNorm-116              [1, 197, 128]             256
#           Linear-117              [1, 197, 512]          66,048
#             GELU-118              [1, 197, 512]               0
#          Dropout-119              [1, 197, 512]               0
#           Linear-120              [1, 197, 128]          65,664
#      FeedForward-121              [1, 197, 128]               0
#          PreNorm-122              [1, 197, 128]               0
#        LayerNorm-123              [1, 197, 128]             256
#           Linear-124              [1, 197, 128]          16,384
#           Linear-125              [1, 197, 128]          16,384
#           Linear-126              [1, 197, 128]          16,384
#          Dropout-127            [1, 8, 197, 64]               0
#           Linear-128              [1, 197, 128]          16,512
# LinformerSelfAttention-129              [1, 197, 128]               0
#          PreNorm-130              [1, 197, 128]               0
#        LayerNorm-131              [1, 197, 128]             256
#           Linear-132              [1, 197, 512]          66,048
#             GELU-133              [1, 197, 512]               0
#          Dropout-134              [1, 197, 512]               0
#           Linear-135              [1, 197, 128]          65,664
#      FeedForward-136              [1, 197, 128]               0
#          PreNorm-137              [1, 197, 128]               0
#        LayerNorm-138              [1, 197, 128]             256
#           Linear-139              [1, 197, 128]          16,384
#           Linear-140              [1, 197, 128]          16,384
#           Linear-141              [1, 197, 128]          16,384
#          Dropout-142            [1, 8, 197, 64]               0
#           Linear-143              [1, 197, 128]          16,512
# LinformerSelfAttention-144              [1, 197, 128]               0
#          PreNorm-145              [1, 197, 128]               0
#        LayerNorm-146              [1, 197, 128]             256
#           Linear-147              [1, 197, 512]          66,048
#             GELU-148              [1, 197, 512]               0
#          Dropout-149              [1, 197, 512]               0
#           Linear-150              [1, 197, 128]          65,664
#      FeedForward-151              [1, 197, 128]               0
#          PreNorm-152              [1, 197, 128]               0
#        LayerNorm-153              [1, 197, 128]             256
#           Linear-154              [1, 197, 128]          16,384
#           Linear-155              [1, 197, 128]          16,384
#           Linear-156              [1, 197, 128]          16,384
#          Dropout-157            [1, 8, 197, 64]               0
#           Linear-158              [1, 197, 128]          16,512
# LinformerSelfAttention-159              [1, 197, 128]               0
#          PreNorm-160              [1, 197, 128]               0
#        LayerNorm-161              [1, 197, 128]             256
#           Linear-162              [1, 197, 512]          66,048
#             GELU-163              [1, 197, 512]               0
#          Dropout-164              [1, 197, 512]               0
#           Linear-165              [1, 197, 128]          65,664
#      FeedForward-166              [1, 197, 128]               0
#          PreNorm-167              [1, 197, 128]               0
#        LayerNorm-168              [1, 197, 128]             256
#           Linear-169              [1, 197, 128]          16,384
#           Linear-170              [1, 197, 128]          16,384
#           Linear-171              [1, 197, 128]          16,384
#          Dropout-172            [1, 8, 197, 64]               0
#           Linear-173              [1, 197, 128]          16,512
# LinformerSelfAttention-174              [1, 197, 128]               0
#          PreNorm-175              [1, 197, 128]               0
#        LayerNorm-176              [1, 197, 128]             256
#           Linear-177              [1, 197, 512]          66,048
#             GELU-178              [1, 197, 512]               0
#          Dropout-179              [1, 197, 512]               0
#           Linear-180              [1, 197, 128]          65,664
#      FeedForward-181              [1, 197, 128]               0
#          PreNorm-182              [1, 197, 128]               0
# SequentialSequence-183              [1, 197, 128]               0
#        Linformer-184              [1, 197, 128]               0
#         Identity-185                   [1, 128]               0
#        LayerNorm-186                   [1, 128]             256
#           Linear-187                     [1, 2]             258
# ================================================================

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