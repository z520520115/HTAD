import torch
import torch.nn as nn
import torch.nn as tnn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import __future__

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
                    print(pic.shape, pic.ndim) # torch.Size([1, 360, 480]) 3
                    pic = self.transform(pic)
                    print(pic.shape, pic.ndim) # torch.Size([1, 224, 224]) 3
                    # print(target)
                    if target == '0':
                        label = torch.tensor(0)
                    else:
                        label = torch.tensor(1)

                    information = {
                        'image': pic,
                        'target': label
                    }

                    self.set.append(information)
                    print(information)

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

    train_loader = DataLoader(training_set, batch_size = 2)
    test_loader = DataLoader(test_set, batch_size = 2)

    print(len(train_loader))
    print(len(test_loader))
    # 查看train_loader中查看数据
    # i1,l1 = next(iter(train_loader))

    return train_loader, test_loader

# Current_Frame Data Loading
class cuf_dataset(Dataset):
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

                    information = {
                        'image': pic,
                        'target': label
                    }

                    self.set.append(information)

    def __getitem__(self, index):
        # print(self.set[index])
        return self.set[index]

    def __len__(self):
        return len(self.set)

def load_cuf_datasets():
    tra_train_root = './simple_dataset/current_fream_0615/train'
    tra_test_root = './simple_dataset/current_fream_0615/test'

    all_type = ["0", "1"]

    training_set = cuf_dataset(root = tra_train_root, all_type = all_type)
    test_set = cuf_dataset(root = tra_test_root, all_type = all_type)

    train_loader = DataLoader(training_set, batch_size = 5)
    test_loader = DataLoader(test_set, batch_size = 5)

    # print(len(train_loader))
    # print(len(test_loader))

    return train_loader, test_loader

# Transformer Module
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

class Transformer(nn.Module):
    def __init__(self, encoder, src_pad_idx, trg_pad_idx, device):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention

# CNN Module
def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer

class VGG16(tnn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7 * 7 * 512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = tnn.Linear(4096, n_classes)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        vgg16_features = self.layer5(x)
        x = vgg16_features.view(x.size(0), -1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        return vgg16_features, x

def change_dim(pic):
    '''change dimension from [C H W] to [H W C]'''
    return pic.permute(1, 2, 0)

def main():
    model1 = VGG16(n_classes=25)
    train_loader, test_loader = load_cuf_datasets()

    optimizer = optim.SGD(model1.parameters(), lr=1e-4, momentum=0.5)
    loss_fn = nn.CrossEntropyLoss()

    loss_all = []

    for epoch in range(2):
        print(f'\n-----------epoch {epoch}-----------')
        loss = train(model1, train_loader, optimizer, loss_fn, epoch=epoch)
        loss_all.append(loss)
        test(model1, test_loader)

    plt.plot(loss_all)
    plt.savefig(f"model_weights/{model1.__class__.__name__}.png")
    plt.show()
    plt.close()

    torch.save(model1.state_dict(), f"model_weights/{model1.__class__.__name__}.pth")
    print("Saved PyTorch Model State to model.pth")

    model1 = VGG16(n_classes=25)
    model1.load_state_dict(torch.load(f"model_weights/{model1.__class__.__name__}.pth"))
    labels = {0: 'No Accident', 1: 'Accident'}
    model1.eval()
    plt.figure(figsize=(8, 4))
    for id, data in enumerate(test_loader):

        if isinstance(data, list):
            image = data[0].type(torch.FloatTensor)
            # target = data[1].to(device)
        elif isinstance(data, dict):
            image = data['image'].type(torch.FloatTensor)
            # target = data['target'].to(device)
        else:
            raise TypeError

        plt.title("image-show")
        with torch.no_grad():
            vgg16_features, output = model1(image)
            # output = nn.Softmax(dim=1)(model1(image))
            # pred = output.argmax(dim=1).cpu().numpy()
            output = nn.Softmax(dim=1)(output)
            pred = output.argmax(dim=1).numpy()

            plt.ion()
            for i in range(1, 5):
                plt.subplot(1, 4, i)
                plt.title(labels[pred[i - 1]])
                plt.imshow(change_dim(image[i - 1].cpu()))
            plt.pause(3)
            plt.show()

# Only VGG16
def train(model, train_loader, optimizer, loss_fn, epoch):
    model.train()

    loss_total = 0
    for _, data in enumerate(train_loader):

        if isinstance(data, list):
            image = data[0].type(torch.FloatTensor)
            target = data[1]
        elif isinstance(data, dict):
            image = data['image'].type(torch.FloatTensor)
            target = data['target']
        else:
            print(type(data))
            raise TypeError
        # print(target)
        optimizer.zero_grad()

        vgg16_features, output = model(image)
        # print(output)

        loss = loss_fn(output, target)
        loss_total += loss.item()

        loss.backward()
        optimizer.step()
    # exit(0)
    print(f'{round(loss_total, 2)} in epoch {epoch}')
    return loss_total

# Only VGG16
def test(model, test_loader):
    model.eval()
    correct = 0

    for _, data in enumerate(test_loader):

        if isinstance(data, list):
            image = data[0].type(torch.FloatTensor)
            target = data[1]
        elif isinstance(data, dict):
            image = data['image'].type(torch.FloatTensor)
            target = data['target']
        else:
            raise TypeError

        with torch.no_grad():
            vgg16_features, output = model(image)
            pred = nn.Softmax(dim=1)(output)

        correct += (pred.argmax(1) == target).type(torch.float).sum().item()

    print(f'accurency = {correct}/{len(test_loader) * 4} = {correct / len(test_loader) / 4}')

if __name__ == "__main__":
    # main()
    load_tra_datasets()