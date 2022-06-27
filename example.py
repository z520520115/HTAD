import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()

        self.linear = nn.Linear(30, 10)

    def forward(self, x):
        return self.linear(x)


class model2(nn.Module):
    def __init__(self):
        super(model2, self).__init__()

        self.linear = nn.Linear(30, 10)

    def forward(self, x):
        return self.linear(x)


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.model1 = model1()
        self.model2 = model2()

        self.linear = nn.Linear(20, 2)

    def forward(self, x, y):
        x1 = self.model1(x)
        x2 = self.model2(y)

        x3 = torch.softmax(self.linear(torch.cat(x1, x2)), dim=-1)
        return x3

sss = model()
print(sss)

t1 = torch.from_numpy(np.zeros([12, 3, 256, 256])).float()
t2 = torch.from_numpy(np.zeros([12, 3, 256, 256])).float()
t3 = torch.from_numpy(np.zeros([12,])).long()

train_loader = DataLoader(TensorDataset(t1, t2, t3), batch_size=4, shuffle=True)

lossfunction = nn.CrossEntropyLoss()
optim = torch.optim.Adam()

for i, batch_data in enumerate(train_loader):
    t_input = batch_data[0]
    c_input = batch_data[1]
    target = batch_data[2]

    output = model(t_input, c_input)
    loss = lossfunction(output, target)
    loss.bachward()
    optim.step()