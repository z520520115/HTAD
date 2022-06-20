import os, torch, glob
import numpy as np
from torch.autograd import Variable
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import shutil
from vgg16 import *

features_dir = '../vgg/features/'


def extractor(img_path, saved_path, net, use_gpu):
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    img = Image.open(img_path)
    img = transform(img)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = tuple(net(t).cpu() for t in x)
    # y = net(x).cpu()
    y = np.array(y)
    # print(y)
    # y = y.data.numpy()
    y = np.around(y, decimals=2)

    np.savetxt(saved_path, y, delimiter='\n')

if __name__ == '__main__':
    files_list = []
    files_list.extend(glob.glob(r"../vgg/train/*.jpg"))

    use_gpu = torch.cuda.is_available()
    vgg16_feature_extractor = VGG16(n_classes=25)  # 要提前知道你使用的网络在全连接层的输入维度即特征维度
    if use_gpu:
        vgg16_feature_extractor = vgg16_feature_extractor.cuda()
        vgg16_feature_extractor = nn.DataParallel(vgg16_feature_extractor)

    pretrained_dict = torch.load(r'../vgg/vgg16.pth')
    model_dict = vgg16_feature_extractor.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(pretrained_dict)
    # del pretrained_dict["module.classifier.weight"]
    # del pretrained_dict["module.classifier.bias"]
    model_dict.update(pretrained_dict)
    vgg16_feature_extractor.load_state_dict(model_dict)
    vgg16_feature_extractor.classifier = nn.Linear(25, 25) # 全连接层设置为25*25 并将weight设置为单位矩阵, 这样输出就是特征
    torch.nn.init.eye_(vgg16_feature_extractor.classifier.weight)

    vgg16_feature_extractor.eval()
    for param in vgg16_feature_extractor.parameters():
        param.requires_grad = False

    for x_path in files_list:
        print(x_path)
        fx_path = os.path.join(features_dir, x_path[13:-4] + '.txt')
        extractor(x_path, fx_path, vgg16_feature_extractor, use_gpu)
