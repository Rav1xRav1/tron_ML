import numpy as np

import torch

from model import Model


class Brain:
    def __init__(self, net, is_gpu, gamma=0.9, r=0.99, lr=0.01, debug=False, device="cpu"):
        self.n_action = 4  # 行動の数

        self.net: Model = net  # ニューラルネットワークのモデル
        self.is_gpu = is_gpu  # GPUを使うかどうか
        self.device = device

        if self.is_gpu:
            self.net.to(self.device)  # GPU対応

        self.eps = 1.0  # ε
        self.gamma = gamma  # 割引率
        self.r = r  # εの減衰率
        self.lr = lr  # 学習係数

        self.debug: bool = debug

    def train(self, states, next_states, action, reward, is_finished):  # ニューラルネットワークを訓練

        states = torch.from_numpy(states).float()  # テンソルに変換
        if next_states is not None: next_states = torch.from_numpy(next_states).float()  # テンソルに変換

        if self.is_gpu:
            states = states.to(self.device)

            if next_states is not None:
                next_states = next_states.to(self.device)

        self.net.eval()  # 評価モード
        if next_states is not None: next_q = self.net(next_states)  # 次の場面をnnに通して値をとる
        self.net.train()  # 訓練モード
        q = self.net(states)  # 今の場面をnnに通して値をとる

        t = q.clone().detach()  # コピーして勾配情報を消す
        if is_finished:
            t[action] = reward  # エピソード終了時の正解は、報酬のみ
        else:
            t[action] = reward + self.gamma * np.max(next_q.detach().cpu().numpy(), axis=0)

        self.net.optimizer.zero_grad()  # 勾配を初期化
        loss = self.net.criterion(q, t)  # 損失を計算
        loss.backward()  # 逆伝播で勾配を最適化
        self.net.optimizer.step()  # 最適化

    def get_action(self, states):  # 行動を取得
        states = torch.from_numpy(states).float()
        if self.is_gpu:
            states = states.to(self.device)  # GPU対応

        if np.random.rand() < self.eps:  # ランダムな行動
            action = np.random.randint(self.n_action)

        else:  # Q値の高い行動を選択
            self.net.eval()  # 評価モード
            q = self.net(states)
            # 行列の一番値が大きい要素のインデックスを返す
            action = np.argmax(q.detach().cpu().numpy(), axis=0)

        if self.eps > 0.1:  # εの下限
            self.eps *= self.r
        return action
