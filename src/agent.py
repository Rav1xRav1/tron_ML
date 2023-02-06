import numpy as np
from random import randint

from tron import Tron


class Agent:
    def __init__(self, brain, env):
        # ゲーム環境
        self.env: Tron = env

        self.brain = brain

        self.game_count: int = 0
        self.win_count: int = 0
        self.loss_count: int = 0
        self.miss_action: int = 0
        self.can_putted: int = 0

        self.reset()

    def reset(self):
        self.env.board_reset()  # ボード情報リセット

    def counter_reset(self):
        self.game_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.miss_action = 0
        self.can_putted = 0

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

                    can_put = False
                    while not can_put:
                        if debug: print("one action:", action)
                        # 駒が置けたならば
                        if self.env.put_one_koma(put_posi=action):
                            self.can_putted += 1
                            now_states = np.array(self.env.get_input_info())
                            # print(now_states[0])
                            past_states = np.array(self.env.get_memorize_board_info())
                            if debug: self.env.print_board()  # ボード表示
                            if debug: print("正しい場所に駒を置けました。")
                            # if debug: print("この状態から", past_states)
                            # if debug: print("この状態に変化", now_states)
                            # if debug: print("価値は0")
                            self.brain.train(past_states, now_states, action, reward=0, terminal=False)
                            self.env.change_turn()  # 順番を交代
                            can_put = True  # 駒が置けました
                            # self.env.print_board()
                        else:  # 駒が置けないなら
                            # if debug: self.env.print_board()
                            # if debug: print("正しい場所に駒を置けませんでした。")
                            self.loss_count += 1
                            # if debug: print("この状態での", now_states)
                            # if debug: print("このアクションが原因です", action)
                            self.brain.train(now_states, None, action, reward=-1, terminal=True)
                            self.reset()  # 試合終了ボード初期化
                            done = True  # 処理終了
                            can_put = True
                            continue
                            # action = self.brain.get_action(now_states)  # 次の一手を決定する
                            # if debug: print("処理のやり直し")
                            # self.miss_action += 1

                else:  # False: 動けない
                    if debug: print("駒が動けませんでした。")
                    self.loss_count += 1
                    # print("この状態での", past_states)
                    # print("この行動が原因でした", self.env.get_before_action(1))
                    self.brain.train(past_states, None, self.env.get_before_action(1), reward=-1, terminal=True)
                    # print(past_states)
                    # print(self.env.get_before_action(1))
                    self.reset()  # 試合終了ボード初期化
                    done = True  # 処理終了
                    continue
            else:
                if self.env.can_move():
                    # action = randint(0, 16 * 20 - 1)
                    action = -1

                    while not self.env.put_two_koma(action):
                        direction = randint(0, 3)
                        if direction == 0:
                            action = self.env.get_enemy_cell() + 1
                        elif direction == 1:
                            action = self.env.get_enemy_cell() + self.env.MAX_COLUMN
                        elif direction == 2:
                            action = self.env.get_enemy_cell() - 1
                        elif direction == 3:
                            action = self.env.get_enemy_cell() - self.env.MAX_COLUMN

                    if debug: print("two action:", action)
                    if debug: self.env.print_board()
                    self.env.change_turn()  # 順番を交代
                else:
                    if debug: print("two loss")
                    self.win_count += 1  # クライアント1の勝利
                    # print("この状態で", past_states)
                    # print("この行動をすると", self.env.get_before_action(1))
                    # print("この状態になります", now_states)
                    self.brain.train(past_states, now_states, self.env.get_before_action(1), reward=1, terminal=True)
                    self.reset()
                    done = True  # 処理終了
                    continue
