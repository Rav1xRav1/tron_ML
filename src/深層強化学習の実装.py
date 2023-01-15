import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from tron_env import Tron_Env


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10 * 10 + 8 + 8 + 1 + 1, 128)  # 全結合層
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 96)
        self.fc4 = nn.Linear(96, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class Brain:
    def __init__(self, net, loss_fnc, optimizer, is_gpu, gamma=0.9, r=0.99, lr=0.01):
        self.n_state = 5 * 5  # 状態の数
        self.n_action = 4  # 行動の数

        self.net = net  # ニューラルネットワークのモデル
        self.loss_fnc = loss_fnc  # 誤差関数
        self.optimizer = optimizer  # 最適化アルゴリズム
        self.is_gpu = is_gpu  # GPUを使うかどうか
        if self.is_gpu:
            self.net.cuda()  # GPU対応

        self.eps = 1.0  # ε
        self.gamma = gamma  # 割引率
        self.r = r  # εの減衰率
        self.lr = lr  # 学習係数

    def train(self, states, next_states, action, reward, terminal):  # ニューラルネットワークを訓練

        states = torch.from_numpy(states).float()  # テンソルに変換
        next_states = torch.from_numpy(next_states).float()  # テンソルに変換
        if self.is_gpu:
            states, next_states = states.cuda(), next_states.cuda()  # GPU対応

        self.net.eval()  # 評価モード
        next_q = self.net.forward(next_states)  # 次の場面をnnに通して値をとる
        self.net.train()  # 訓練モード
        q = self.net.forward(states)  # 今の場面をnnに通して値をとる

        t = q.clone().detach()  # コピーして勾配情報を消す
        if terminal:
            t[:, action] = reward  # エピソード終了時の正解は、報酬のみ
        else:
            t[:, action] = reward + self.gamma * np.max(next_q.detach().cpu().numpy(), axis=1)[0]

        loss = self.loss_fnc(q, t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, states):  # 行動を取得
        states = torch.from_numpy(states).float()
        if self.is_gpu:
            states = states.cuda()  # GPU対応

        if np.random.rand() < self.eps:  # ランダムな行動
            action = np.random.randint(self.n_action)
        else:  # Q値の高い行動を選択
            q = self.net.forward(states)
            action = np.argmax(q.detach().cpu().numpy(), axis=1)[0]
        if self.eps > 0.1:  # εの下限
            self.eps *= self.r
        return action


class Agent:
    def __init__(self, brain: Brian, env: Tron_Env):
        # ゲーム環境
        self.env: Tron_Env = env

        self.brain: Brain = brain

    def reset(self):
        self.env.board_reset()  # ボード情報リセット

    def play(self):
        """
        ゲームを一試合プレイする
        """
        done = False  # True: 試合終了 False: 試合継続

        while not done:
            if self.env.check_turn() == self.env.one_client_koma:
                states = np.array([self.env.get_board_info()])  # ボード情報をndarrayに変換

                reward = 0  # 報酬
                terminal = False  # 終了判定
                action = None  # 次の手

                # すでにゲームが終わっていないか確認する
                if self.env.can_move():  # 自分自身が動けるか
                    action = self.brian.get_action(states)  # 次の一手を決定する
                    print("one action:", action)

                    if self.env.put_one_koma(action=action):
                        # self.env.print_board()
                        print("AI loss")
                        self.brain.train(np.concatenate([self.env.memorize_game_board, ]))
                else:  # False: 動けない
                    reward = -1
                    terminal = True
                    self.brain.train(np.concatenate([states, np.array([[self.env.one_posi]])], 1),
                                     np.concatenate([np.array([self.env.get_memorize_board_info()]),
                                                     np.array([[self.env.one_posi]])], 1), action, -1, terminal)




class Environment:
    def __init__(self):
        # 盤の大きさ
        self.MAX_LINE = 5
        self.MAX_COLUMN = 5

        self.blank: int = 0  # 空白のマス
        self.one_client_koma: int = 1  # クライアント1の駒
        self.two_client_koma: int = -1  # クライアント2の駒
        self.obstacle: int = 2  # 障害物 テストコードでは使用しない

        self.order_of_koma = self.one_client_koma  # 駒の順番を記憶する

        self.game_board: list = [self.blank] * (self.MAX_LINE * self.MAX_COLUMN)  # ゲームボード情報をクリア

    def check_turn(self) -> int:
        return self.order_of_koma  # 現在誰の順番かを返す

    def change_turn(self) -> None:  # 駒の順番を変更する
        if self.check_turn() == self.one_client_koma:
            self.order_of_koma = self.two_client_koma
        else:
            self.order_of_koma = self.one_client_koma

    def can_move(self, board, posi) -> bool:
        # 各方向に対して検査
        if (posi - self.MAX_COLUMN) >= 0 and board[posi - self.MAX_COLUMN] == self.blank:
            return True
        elif (posi + self.MAX_COLUMN) < (self.MAX_COLUMN * self.MAX_LINE) and board[
            posi + self.MAX_COLUMN] == self.blank:
            return True
        elif (posi % self.MAX_COLUMN) != 0 and board[posi - 1] == self.blank:
            return True
        elif ((posi + 1) % self.MAX_COLUMN) != 0 and board[posi + 1] == self.blank:
            return True
        else:
            # 移動不可
            return False

    def can_put(self, posi, action) -> int:
        # 各方向に対して検査
        if action == 0 and ((posi + 1) % self.MAX_COLUMN) != 0 and self.game_board[posi + 1] == self.blank:
            return posi + 1
        elif action == 1 and (posi + self.MAX_COLUMN) < (self.MAX_COLUMN * self.MAX_LINE) and self.game_board[
            posi + self.MAX_COLUMN] == self.blank:
            return posi + self.MAX_COLUMN
        elif action == 2 and (posi % self.MAX_COLUMN) != 0 and board[posi - 1] == self.blank:
            return posi - 1
        elif action == 3 and (posi - self.MAX_COLUMN) >= 0 and board[posi - self.MAX_COLUMN] == self.blank:
            return posi + self.MAX_COLUMN
        else:
            return posi


net = Net()

loss_fnc = nn.MSELoss()  # 誤差関数
optimizer = optim.RMSprop(net.parameters())  # 最適化アルゴリズム
is_gpu = False  # GPUを使用するか否か

brain = Brain(net, loss_fnc, optimizer, is_gpu)

agent = Agent(brain, Tron_Env(max_line=10, max_column=10, one_start_posi=42, two_start_posi=47))
