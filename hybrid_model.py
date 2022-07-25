import os
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

from vgg16_model import VGG16
from vision_transformer.vit_model import VisionTransformer
from vision_transformer.vit_model import vit_base_patch16_224_in21k as create_model
from vision_transformer.hybrid_model_utils import read_split_data, train_one_epoch, evaluate

class hybrid_model(nn.Module):
    def __init__(self, VisionTransformer, VGG16):
        super(hybrid_model, self).__init__()
        self.model1 = VisionTransformer()
        self.linear1 = nn.Linear(1000, 500)
        self.model2 = VGG16()
        self.linear2 = nn.Linear(1000, 500)
        self.linear3 = nn.Linear(1000, 2)
        # self.model1_linear = nn.Linear(100 * 10, 25)

    def forward(self, x1, x2):
        x1 = self.model1(x1)
        x1 = self.linear1(x1)
        _, x2 = self.model2(x2)
        x2 = self.linear2(x2)
        x3 = torch.cat((x1, x2), dim=1)
        x3 = self.linear3(x3)
        x3 = torch.softmax(x3, dim=-1)
        return x3

class VitDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

class VggDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_tras_path, train_tras_label, val_tras_path, val_tras_label = read_split_data(
        r"C:/Users/YIHANG/PycharmProjects/HTAD_dataset/trajectory_mask")
    train_imgs_path, train_imgs_label, val_imgs_path, val_imgs_label = read_split_data(
        r"C:/Users/YIHANG/PycharmProjects/HTAD_dataset/current_frame")

    Vit_data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    Vgg_data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     # transforms.ConvertImageDtype(torch.float64)]),
                                    transforms.ToTensor()]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   # transforms.ConvertImageDtype(torch.float64)]),
                                   transforms.ToTensor()])}
    # 实例化Vit数据集
    Vit_train_dataset = VitDataSet(images_path=train_tras_path,
                              images_class=train_tras_label,
                              transform=Vit_data_transform["train"])
    Vit_val_dataset = VitDataSet(images_path=val_tras_path,
                            images_class=val_tras_label,
                            transform=Vit_data_transform["val"])

    # 实例化Vgg数据集
    Vgg_train_dataset = VggDataSet(images_path=train_imgs_path,
                              images_class=train_imgs_label,
                              transform=Vgg_data_transform["train"])
    Vgg_val_dataset = VggDataSet(images_path=val_tras_path,
                                 images_class=val_imgs_label,
                                 transform=Vgg_data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    Vit_train_loader = DataLoader(Vit_train_dataset,
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  num_workers=nw,
                                  )#collate_fn=Vit_train_dataset.collate_fn)

    Vit_val_loader = DataLoader(Vit_val_dataset,
                                batch_size=batch_size,
                                pin_memory=True,
                                num_workers=nw,
                                )#collate_fn=Vit_val_dataset.collate_fn)

    Vgg_train_loader = DataLoader(Vgg_train_dataset,
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  num_workers=nw,
                                  )#collate_fn=Vgg_train_dataset.collate_fn)

    Vgg_val_loader = DataLoader(Vgg_val_dataset,
                                batch_size=batch_size,
                                pin_memory=True,
                                num_workers=nw,
                                )#collate_fn=Vgg_val_dataset.collate_fn)


    label, imgs, tras = map(dataloader_sort, [Vgg_train_loader, Vgg_train_loader, Vit_train_loader], [1, 0, 0], [True, False, False])
    label_v, imgs_v, tras_v = map(dataloader_sort, [Vgg_val_loader, Vgg_val_loader, Vit_val_loader], [1, 0, 0], [True, False, False])

    # 整合两种数据集 DataLoader[0]为轨迹, [1]当前帧, [2]标签 (数据集做好的情况下两者标签应为一致)
    train_loader = DataLoader(TensorDataset(tras, imgs, label),
                              batch_size=batch_size,
                              pin_memory=True,
                              num_workers=nw)

    val_loader = DataLoader(TensorDataset(tras_v, imgs_v, label_v),
                              batch_size=batch_size,
                              pin_memory=True,
                              num_workers=nw)

    # VisionTransformer = create_model(num_classes=2, has_logits=False).to(device)
    model = hybrid_model(VisionTransformer, VGG16)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)


        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # torch.save(model.state_dict(), "vision_transformer/weights/hybrid_model-{}.pth".format(epoch))
        torch.save(model, "vision_transformer/weights/hybrid_model.pkl")
def dataloader_sort(loader, index, is_label):
    if not is_label:
        return torch.from_numpy(np.array([i[index][0].numpy().tolist() for i in iter(loader)])).float()
    else:
        return torch.from_numpy(np.array([i[index][0].numpy().tolist() for i in iter(loader)])).long()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)