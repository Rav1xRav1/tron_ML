from torch import nn
from torch import optim


class Model(nn.Module):
    def __init__(self, height=10, width=10, input_channel=1, output_size=4):
        super().__init__()

        self.relu = nn.ReLU()  # 活性化関数

        self.conv1 = nn.Conv2d(input_channel, 32, 4)  # 畳み込み層
        self.conv2 = nn.Conv2d(32, 64, 4)

        # self.pool = nn.MaxPool2d(3, 1)  # プーリング層

        self.fc1 = nn.Linear(1024, 1024)  # 全結合層
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_size)

        # self.softmax = nn.Softmax(dim=0)  # ソフトマックス関数

        # 損失関数と最適化関数
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())
        # self.optimizer = optim.RMSprop(self.parameters())

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = x.view(-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        # x = self.softmax(x)

        return x
