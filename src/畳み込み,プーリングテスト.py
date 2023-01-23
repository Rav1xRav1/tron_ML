import numpy as np
import torch
import numpy

import torch.nn as nn
from torch.nn import functional as F


x = torch.ones(5, 5, 5)
print(x)
print("0", x.size())

relu = nn.ReLU()
# pool = nn.MaxPool2d(2, stride=1)

conv1 = nn.Conv2d(5, 15, 2)
conv2 = nn.Conv2d(15, 30, 2)

fc1 = nn.Linear(270, 135)
fc2 = nn.Linear(135, 25)

softmax = nn.Softmax(dim=0)

x = conv1(x)
print("1", x.size())
x = relu(x)
print("2", x.size())
# x = pool(x)
# print("3", x.size())
# print(x)
x = conv2(x)
print("4", x.size())
x = relu(x)
print("5", x.size())
# x = pool(x)
# print("6", x.size())
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
print(torch.argmax(x))
print(np.max(x.detach().numpy()))
