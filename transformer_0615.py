import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
import math
import time
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torch.utils import data
from torchvision import transforms
from PIL import Image

seed = 42
random.seed(seed)
os.environ['PYTHONSHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# Trajectory Data Loading
class tra_dataset(Dataset):
    def __init__(self, root, all_type) -> None:
        self.root = root
        self.all_type = all_type
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ConvertImageDtype(torch.float64),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.set = []

        for root, dirs, files in os.walk(self.root):
            target = root.split('\\')[-1]
            if target in self.all_type:
                for file in files:
                    pic = read_image(os.path.join(root, file))
                    pic = self.transform(pic)
                    # print(target)
                    if target == '0':
                        label = torch.tensor(0)
                    else:
                        label = torch.tensor(1)
                    information_1 = {
                        'image': pic,
                        'target': label
                    }
                    self.set.append(information_1)
                    
    def __getitem__(self, index):
        print(self.set[index])
        return self.set[index]

    def __len__(self):
        return len(self.set)

def load_tra_datasets():
    tra_train_root = './simple_dataset/trajectory_mask_0615/train'
    tra_test_root = './simple_dataset/trajectory_mask_0615/test'

    all_type = ["0", "1"]

    training_set = tra_dataset(root = tra_train_root, all_type = all_type)
    test_set = tra_dataset(root = tra_test_root, all_type = all_type)

    train_loader = DataLoader(training_set, batch_size = 1)
    test_loader = DataLoader(test_set, batch_size = 1)

    # 查看train_loader中查看数据
    batch = iter(train_loader)
    images, labels = batch.next()
    # print(images)

    return  train_loader, test_loader

# Transformer Class
class PatchEmbed(nn.Module):
    def __init__(self, img_size = 224, patch_size = 16, in_c = 3, embed_dim = 768, norm_layer = None):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forwad(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flattent: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C , HW] -> [B, HW, C]
        x = self.proj(x). flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 100):
        super(Encoder, self).__init__()

        self.device = device
        self.img_embedding = nn.Embedding(input_dim, hid_dim) # an Embedding module containing input_dim tensors of size hid_dim
        self.pos_embedding = nn.Embedding(max_length, hid_dim) # max_length x hid_dim

        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def Forward(self, src, src_mask):
        # src =[batch size, src len]
        # src_mask = [batch size, 1, 1, src len]
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # pos = [batch size, src len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # src = [batch size, src len, hid dim]
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]
        for layer in self.layers:
            src = layer(src, src_mask)

        return src

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super(EncoderLayer, self).__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        # src = [batch size, src len, hid dim]
        # src mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super(MultiHeadAttentionLayer, self).__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # attention = [batch size, n heads, query len, key len]
        attention = torch.softmax(energy, dim = -1)

        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch size, query len , hid dim]

        x = self.fc_o(x)

        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super(PositionwiseFeedforwardLayer, self).__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]
        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x

load_tra_datasets()