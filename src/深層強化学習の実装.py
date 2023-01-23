import numpy as np
from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from tron_env import Tron_Env


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        """
        self.fc1 = nn.Linear(10 * 10 + 8 + 8 + 1 + 1, 128)  # 全結合層
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 96)
        self.fc4 = nn.Linear(96, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 4)
        """

        self.relu = nn.ReLU()  # 活性化関数
        # self.pool = nn.MaxPool2d(2, stride=2)  # プーリング層

        self.conv1 = nn.Conv2d(5, 15, 2)  # 畳み込み層
        self.conv2 = nn.Conv2d(15, 30, 2)

        self.fc1 = nn.Linear(270, 128)  # 全結合層
        self.fc2 = nn.Linear(128, 62)
        self.fc3 = nn.Linear(62, 4)

        self.softmax = nn.Softmax(dim=0)  # ソフトマックス関数

    def forward(self, x):
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        """

        x = self.conv1(x)
        x = self.relu(x)
        # x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        # x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)

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
        # print(next_states, next_states.size())
        if self.is_gpu:
            states, next_states = states.cuda(), next_states.cuda()  # GPU対応

        self.net.eval()  # 評価モード
        next_q = self.net.forward(next_states)  # 次の場面をnnに通して値をとる
        self.net.train()  # 訓練モード
        # print("states", states)
        q = self.net.forward(states)  # 今の場面をnnに通して値をとる
        # print("next_q", next_q)
        # print("q", q)

        t = q.clone().detach()  # コピーして勾配情報を消す
        # print("t", t)
        if terminal:
            t[action] = reward  # エピソード終了時の正解は、報酬のみ
        else:
            # print(next_q.detach().cpu().numpy())
            # print(np.max(next_q.detach().cpu().numpy(), axis=0))
            t[action] = reward + self.gamma * np.max(next_q.detach().cpu().numpy(), axis=0)

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
            # softmaxを嚙ませたうえで行列の一番値が大きい要素のインデックスを返す
            action = np.argmax(q.detach().cpu().numpy(), axis=0)
            # print(action)
        if self.eps > 0.1:  # εの下限
            self.eps *= self.r
        return action


class Agent:
    def __init__(self, brain: Brain, env: Tron_Env):
        # ゲーム環境
        self.env: Tron_Env = env

        self.brain: Brain = brain

        self.game_count: int = 0
        self.win_count: int = 0
        self.loss_count: int = 0

    def reset(self):
        self.env.board_reset()  # ボード情報リセット

    def counter_reset(self):
        self.game_count = 0
        self.win_count = 0
        self.loss_count = 0

    def play(self, debug: bool = False):
        """
        ゲームを一試合プレイする
        """
        done = False  # True: 試合終了 False: 試合継続
        self.game_count += 1

        while not done:
            if debug: print("今回の順番は:", self.env.check_turn())
            now_states = np.array(self.env.get_input_info())
            past_states = np.array(self.env.get_memorize_board_info())
            if self.env.check_turn() == self.env.one_client_koma:

                # すでにゲームが終わっていないか確認する
                if self.env.can_move():  # 自分自身が動けるか
                    action = self.brain.get_action(now_states)  # 次の一手を決定する
                    if debug: print("one action:", action)

                    # 駒が置けたならば
                    if self.env.put_one_koma(action=action):
                        now_states = np.array(self.env.get_input_info())
                        past_states = np.array(self.env.get_memorize_board_info())
                        # if debug: print("now", now_states)
                        # if debug: print("past", past_states)
                        if debug: self.env.print_board()  # ボード表示
                        if debug: print("正しい場所に駒を置けました。")
                        self.brain.train(past_states, now_states, action, reward=0, terminal=False)
                        self.env.change_turn()  # 順番を交代
                    else:  # 駒が置けないなら
                        if debug: self.env.print_board()
                        if debug: print("正しい場所に駒を置けませんでした。")
                        self.loss_count += 1
                        self.brain.train(past_states, now_states, action, reward=-1, terminal=True)
                        self.env.board_reset()  # 試合終了ボード初期化
                        done = True  # 処理終了
                        continue
                else:  # False: 動けない
                    if debug: print("駒が動けませんでした。")
                    self.loss_count += 1
                    self.brain.train(past_states, now_states, self.env.get_before_action(1), reward=-1, terminal=True)
                    self.env.board_reset()  # 試合終了ボード初期化
                    done = True  # 処理終了
                    continue
            else:
                if self.env.can_move():
                    action = randint(0, 4)
                    while not self.env.put_two_koma(action):
                        action = randint(0, 4)

                    if debug: print("two action:", action)
                    if debug: self.env.print_board()
                    self.env.change_turn()  # 順番を交代
                else:
                    if debug: print("two loss")
                    self.win_count += 1  # クライアント1の勝利
                    self.brain.train(past_states, now_states, self.env.get_before_action(1), reward=1, terminal=True)
                    self.env.board_reset()
                    done = True  # 処理終了
                    continue


net = Net()

loss_fnc = nn.MSELoss()  # 誤差関数
optimizer = optim.RMSprop(net.parameters())  # 最適化アルゴリズム
is_gpu = False  # GPUを使用するか否か

brain = Brain(net, loss_fnc, optimizer, is_gpu)

agent = Agent(brain, Tron_Env(max_line=5, max_column=5, one_start_posi=11, two_start_posi=13))

for i in range(100000):
    if i % 1000 == 0:
        print(f"{i}回")
        print("game:", agent.game_count)
        print("win:", agent.win_count)
        print("loss:", agent.loss_count, end="\n\n")
        agent.counter_reset()
    agent.play(debug=False)

torch.save(brain.net, f="tron_dqn.pth")
