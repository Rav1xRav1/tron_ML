import torch

import torch.nn as nn
from torch.nn import functional as F


x = torch.rand(5, 20, 20)
print("0", x.size())

relu = nn.ReLU()
pool = nn.MaxPool2d(2, stride=2)

conv1 = nn.Conv2d(5, 16, 3)
conv2 = nn.Conv2d(16, 32, 3)

fc1 = nn.Linear(32*9, 120)
fc2 = nn.Linear(120, 15)

softmax = nn.Softmax(dim=0)

x = conv1(x)
print("1", x.size())
x = relu(x)
print("2", x.size())
x = pool(x)
print("3", x.size())
x = conv2(x)
print("4", x.size())
x = relu(x)
print("5", x.size())
x = pool(x)
print("6", x.size())
x = x.view(x.size()[0], -1)
print("7", x.size())
x = x.view(-1)
print("8", x.size())
x = fc1(x)
print("9", x.size())
x = relu(x)
print("10", x.size())
x = fc2(x)

x = softmax(x)

print(x)
