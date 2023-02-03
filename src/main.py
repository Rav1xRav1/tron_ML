import numpy as np
import datetime as dt
from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from tron import tron
from model import model
from brain import Brain


class Agent:
    def __init__(self, brain: Brain, env: tron):
        # ゲーム環境
        self.env: tron_board = env

        self.brain: Brain = brain

        self.game_count: int = 0
        self.win_count: int = 0
        self.loss_count: int = 0
        self.miss_action: int = 0

    def reset(self):
        self.env.board_reset()  # ボード情報リセット

    def counter_reset(self):
        self.game_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.miss_action = 0

    def play(self, debug: bool = False):
        """
        ゲームを一試合プレイする
        """
        done = False  # True: 試合終了 False: 試合継続
        self.game_count += 1

        while not done:
            if debug: print("今回の順番は:", self.env.check_turn())
            # print(self.env.get_input_info())
            now_states = np.array(self.env.get_input_info())
            past_states = np.array(self.env.get_memorize_board_info())
            if self.env.check_turn() == self.env.one_client_koma:

                # すでにゲームが終わっていないか確認する
                if self.env.can_move():  # 自分自身が動けるか
                    action = self.brain.get_action(now_states)  # 次の一手を決定する

                    can_put = False
                    while not can_put:
                        if debug: print("one action:", action)
                        # 駒が置けたならば
                        # print(action)
                        if self.env.put_one_koma(put_posi=action):
                            # print(action)
                            now_states = np.array(self.env.get_input_info())
                            past_states = np.array(self.env.get_memorize_board_info())
                            if debug: self.env.print_board()  # ボード表示
                            if debug: print("正しい場所に駒を置けました。")
                            # if debug: print("この状態から", past_states)
                            # if debug: print("この状態に変化", now_states)
                            # if debug: print("価値は0")
                            self.brain.train(past_states, now_states, action, reward=1, terminal=False)
                            self.env.change_turn()  # 順番を交代
                            can_put = True  # 駒が置けました
                            # self.env.print_board()
                        else:  # 駒が置けないなら
                            # if debug: self.env.print_board()
                            # if debug: print("正しい場所に駒を置けませんでした。")
                            # self.loss_count += 1
                            # if debug: print("この状態での", now_states)
                            # if debug: print("このアクションが原因です", action)
                            self.brain.train(now_states, None, action, reward=-1, terminal=True)
                            # self.env.board_reset()  # 試合終了ボード初期化
                            # done = True  # 処理終了
                            action = self.brain.get_action(now_states)  # 次の一手を決定する
                            # if debug: print("処理のやり直し")
                            self.miss_action += 1

                else:  # False: 動けない
                    if debug: print("駒が動けませんでした。")
                    self.loss_count += 1
                    # print("この状態での", past_states)
                    # print("この行動が原因でした", self.env.get_before_action(1))
                    self.brain.train(past_states, None, self.env.get_before_action(1), reward=-1, terminal=True)
                    # print(past_states)
                    # print(self.env.get_before_action(1))
                    self.env.board_reset()  # 試合終了ボード初期化
                    done = True  # 処理終了
                    continue
            else:
                if self.env.can_move():
                    action = randint(0, 16 * 20 - 1)
                    while not self.env.put_two_koma(action):
                        action = randint(0, 16 * 20 - 1)

                    if debug: print("two action:", action)
                    if debug: self.env.print_board()
                    self.env.change_turn()  # 順番を交代
                else:
                    if debug: print("two loss")
                    self.win_count += 1  # クライアント1の勝利
                    # print("この状態で", past_states)
                    # print("この行動をすると", self.env.get_before_action(1))
                    # print("この状態になります", now_states)
                    self.brain.train(past_states, now_states, self.env.get_before_action(1), reward=5, terminal=True)
                    self.env.board_reset()
                    done = True  # 処理終了
                    continue


start_time = dt.datetime.now()
print("処理開始:", start_time)

model = model()

is_gpu = False  # GPUを使用するか否か

# model = torch.load("D:/Python/machine_learning/src/tron_dqn.pth")
brain = Brain(model, is_gpu, debug=False)

max_column = 20
max_line = 16

print("クライアント１のスタート位置:", int(max_line / 2 * max_column + max_column / 4))
tron = tron(max_line=max_line, max_column=max_column,
                one_start_posi=int(max_line / 2 * max_column + max_column / 4),
                two_start_posi=int((max_line / 2 + 1) * max_column - (max_column / 4 + 1)))
agent = Agent(brain, env=tron)

# print(torch.argmax(model(torch.tensor(tron.get_input_info()).float())))
print(torch.argmax(model(torch.tensor(tron.get_input_info()).float())))
# exit()

do_play = "y"

while do_play != "n":
    for i in range(10):
        if (i + 1) % 1 == 0:
            print("・-", end="")
        if (i + 1) % 10 == 0:
            print()
            print(f"{i + 1}回")
            print("game:", agent.game_count)
            print("win:", agent.win_count)
            print("loss:", agent.loss_count)
            print("miss:", agent.miss_action, end="\n")
            agent.counter_reset()
            # print("ε:", agent.brain.eps, end="\n\n")
            print(f"{i + 1}回目の終了時刻: {dt.datetime.now()}")
        agent.play(debug=False)
    print("初めに選択すべき値は:", torch.argmax(model(torch.tensor(tron.get_input_info()).float())))
    torch.save(brain.net, f="tron_dqn.pth")
    do_play = input("続けますか？ : ")

torch.save(brain.model, f="tron_dqn.pth")

finish_time = dt.datetime.now()
print("処理終了:", finish_time)
print("実行時間:", finish_time - start_time)
