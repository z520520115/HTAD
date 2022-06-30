import torch
import torch.nn as nn
import torch.nn as tnn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import __future__

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

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
    tra_train_root = '../simple_dataset/current_fream_0615/train'
    tra_test_root = '../simple_dataset/current_fream_0615/test'

    all_type = ["0", "1"]

    training_set = cuf_dataset(root = tra_train_root, all_type = all_type)
    test_set = cuf_dataset(root = tra_test_root, all_type = all_type)

    train_loader = DataLoader(training_set, batch_size = 5)
    test_loader = DataLoader(test_set, batch_size = 5)

    print(len(train_loader))
    print(len(test_loader))

    i1, l1 = next(iter(train_loader))
    print(i1)
    print(l1)

    return train_loader, test_loader

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
        # print("Vgg第八层shape" + str(x.shape))
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
    main()
