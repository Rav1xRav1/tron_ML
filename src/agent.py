import numpy as np
from random import randint

from tron import Tron


class Agent:
    def __init__(self, brain, env):
        # ゲーム環境
        self.tron: Tron = env

        self.brain = brain

        self.game: int = 0
        self.win: int = 0
        self.lose: int = 0

        self.reset()

    def reset(self):
        self.tron.board_reset()  # ボード情報リセット

    def counter_reset(self):
        self.game, self.win, self.lose = 0, 0, 0

    def get_counter(self) -> tuple[int, int, int]:
        return self.game, self.win, self.lose

    def play(self, debug: bool = False):
        """
        ゲームを一試合プレイする
        """
        done = False  # True: 試合終了 False: 試合継続
        self.game += 1

        while not done:
            if debug: print("今回の順番は:", self.tron.check_turn())
            now_states = np.array(self.tron.get_input_info())
            past_states = np.array(self.tron.get_memorize_board_info())
            if self.tron.check_turn() == self.tron.one_client_koma:

                # すでにゲームが終わっていないか確認する
                if self.tron.can_move():  # 自分自身が動けるか
                    action = self.brain.get_action(now_states)  # 次の一手を決定する

                    can_put = False
                    while not can_put:
                        if debug: print("one action:", action)
                        # 駒が置けたならば
                        if self.tron.put_one_koma(direction=action):
                            now_states = np.array(self.tron.get_input_info())
                            past_states = np.array(self.tron.get_memorize_board_info())
                            if debug: self.tron.print_board()  # ボード表示
                            if debug: print("正しい場所に駒を置けました。")
                            if debug: print("この状態から")
                            self.tron.print_board(board=past_states, is_debug=True)
                            if debug: print("この状態に変化")
                            self.tron.print_board(board=now_states, is_debug=True)
                            if debug: print("価値は0")
                            self.brain.train(past_states, now_states, action, reward=0, is_finished=False)
                            self.tron.change_turn()  # 順番を交代
                            can_put = True  # 駒が置けました
                        else:  # 駒が置けないなら
                            if debug: print("正しい場所に駒を置けませんでした。")
                            if debug: print("この状態での")
                            self.tron.print_board(board=now_states, is_debug=True)
                            if debug: print("このアクションが原因です:", action)
                            self.brain.train(now_states, past_states, action, reward=-1, is_finished=True)  # 第二引数意味なし
                            self.reset()  # 試合終了ボード初期化
                            done = True  # 処理終了
                            can_put = True
                            self.lose += 1
                            continue
                            # action = self.brain.get_action(now_states)  # 次の一手を決定する

                else:  # False: 動けない
                    if debug: print("駒が動けませんでした。")
                    print("この状態での")
                    self.tron.print_board(board=past_states, is_debug=True)
                    print("この行動が原因でした", self.env.get_before_action(1))
                    self.brain.train(past_states, past_states, self.tron.memory.get_memory_one_action(), reward=-1, is_finished=True)  # 第二引数意味なし
                    self.reset()  # 試合終了ボード初期化
                    done = True  # 処理終了
                    self.lose += 1
                    continue
            else:
                if self.tron.can_move():
                    action = -1

                    while not self.tron.put_two_koma(action):
                        action = randint(0, 3)

                    if debug: print("two action:", action)
                    if debug: self.tron.print_board()
                    self.tron.change_turn()  # 順番を交代
                else:
                    if debug: print("two loss")
                    # print("この状態で", past_states)
                    # print("この行動をすると", self.env.get_before_action(1))
                    # print("この状態になります", now_states)
                    self.brain.train(past_states, now_states, self.tron.memory.get_memory_one_action(), reward=1, is_finished=True)
                    self.reset()
                    done = True  # 処理終了
                    self.win += 1
                    continue
