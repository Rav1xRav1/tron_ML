from torch import nn
from torch import optim


class model(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()  # 活性化関数

        self.conv1 = nn.Conv2d(3, 6, 2)  # 畳み込み層

        self.fc1 = nn.Linear(1512, 1024)  # 全結合層
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 20*16)

        self.softmax = nn.Softmax(dim=0)  # ソフトマックス関数

        # 損失関数と最適化関数
        self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.parameters())
        self.optimizer = optim.RMSprop(self.parameters())

    def forward(self, x):
        print(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size()[0], -1)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.softmax(x)

        return x
