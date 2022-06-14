import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary

BATCH_SIZE = 100
LEARNING_RATE = 0.01
EPOCH = 1
N_CLASSES = 25

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

trainData = dsets.ImageFolder('../simple_dataset/images/train', transform)
testData = dsets.ImageFolder('../simple_dataset/images/val', transform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)


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

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out


vgg16 = VGG16(n_classes=N_CLASSES).to(device)
# vgg16.cuda()

# Network parameter visualization
# summary(vgg16, (3, 224, 224), batch_size=100)
'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1        [100, 64, 224, 224]           1,792
       BatchNorm2d-2        [100, 64, 224, 224]             128
              ReLU-3        [100, 64, 224, 224]               0
            Conv2d-4        [100, 64, 224, 224]          36,928
       BatchNorm2d-5        [100, 64, 224, 224]             128
              ReLU-6        [100, 64, 224, 224]               0
         MaxPool2d-7        [100, 64, 112, 112]               0
            Conv2d-8       [100, 128, 112, 112]          73,856
       BatchNorm2d-9       [100, 128, 112, 112]             256
             ReLU-10       [100, 128, 112, 112]               0
           Conv2d-11       [100, 128, 112, 112]         147,584
      BatchNorm2d-12       [100, 128, 112, 112]             256
             ReLU-13       [100, 128, 112, 112]               0
        MaxPool2d-14         [100, 128, 56, 56]               0
           Conv2d-15         [100, 256, 56, 56]         295,168
      BatchNorm2d-16         [100, 256, 56, 56]             512
             ReLU-17         [100, 256, 56, 56]               0
           Conv2d-18         [100, 256, 56, 56]         590,080
      BatchNorm2d-19         [100, 256, 56, 56]             512
             ReLU-20         [100, 256, 56, 56]               0
           Conv2d-21         [100, 256, 56, 56]         590,080
      BatchNorm2d-22         [100, 256, 56, 56]             512
             ReLU-23         [100, 256, 56, 56]               0
        MaxPool2d-24         [100, 256, 28, 28]               0
           Conv2d-25         [100, 512, 28, 28]       1,180,160
      BatchNorm2d-26         [100, 512, 28, 28]           1,024
             ReLU-27         [100, 512, 28, 28]               0
           Conv2d-28         [100, 512, 28, 28]       2,359,808
      BatchNorm2d-29         [100, 512, 28, 28]           1,024
             ReLU-30         [100, 512, 28, 28]               0
           Conv2d-31         [100, 512, 28, 28]       2,359,808
      BatchNorm2d-32         [100, 512, 28, 28]           1,024
             ReLU-33         [100, 512, 28, 28]               0
        MaxPool2d-34         [100, 512, 14, 14]               0
           Conv2d-35         [100, 512, 14, 14]       2,359,808
      BatchNorm2d-36         [100, 512, 14, 14]           1,024
             ReLU-37         [100, 512, 14, 14]               0
           Conv2d-38         [100, 512, 14, 14]       2,359,808
      BatchNorm2d-39         [100, 512, 14, 14]           1,024
             ReLU-40         [100, 512, 14, 14]               0
           Conv2d-41         [100, 512, 14, 14]       2,359,808
      BatchNorm2d-42         [100, 512, 14, 14]           1,024
             ReLU-43         [100, 512, 14, 14]               0
        MaxPool2d-44           [100, 512, 7, 7]               0
           Linear-45                [100, 4096]     102,764,544
      BatchNorm1d-46                [100, 4096]           8,192
             ReLU-47                [100, 4096]               0
           Linear-48                [100, 4096]      16,781,312
      BatchNorm1d-49                [100, 4096]           8,192
             ReLU-50                [100, 4096]               0
           Linear-51                  [100, 25]         102,425
================================================================
Total params: 134,387,801
Trainable params: 134,387,801
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 57.42
Forward/backward pass size (MB): 573.84
Params size (MB): 512.65
Estimated Total Size (MB): 1143.91
----------------------------------------------------------------
'''

# Loss, Optimizer & Scheduler
cost = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# Train the model
for epoch in range(EPOCH):

    avg_loss = 0
    cnt = 0
    for images, labels in trainLoader:
        # images = images.cuda()
        # labels = labels.cuda()
        images = images.to(device)
        labels = labels.to(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        _, outputs = vgg16(images)
        loss = cost(outputs, labels)
        avg_loss += loss.data
        cnt += 1
        print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss / cnt))
        loss.backward()
        optimizer.step()
    scheduler.step(avg_loss)

# Test the model
vgg16.eval()
correct = 0
total = 0

for images, labels in testLoader:
    # images = images.cuda()
    images = images.to(device)
    _, outputs = vgg16(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    print(predicted, labels, correct, total)
    print("avg acc: %f" % (100 * correct / total))

# Save the Trained Model
torch.save(vgg16.state_dict(), 'vgg16.pth')