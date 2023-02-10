import numpy as np
from random import randint

from tron import Tron


class Agent:
    def __init__(self, brain, env, debug: bool = False):
        # ゲーム環境
        self.tron: Tron = env

        self.brain = brain

        self.debug: bool = debug

        self.win: int = 0

        self.reset()

    def reset(self):
        self.tron.board_reset()  # ボード情報リセット

    def reset_counter(self):
        self.win = 0

    def get_counter(self) -> int:
        return self.win

    def play(self):
        """
        ゲームを一試合プレイする
        """
        done = False  # True: 試合終了 False: 試合継続

        while not done:
            if self.debug: print("今回の順番は:", self.tron.check_turn())
            now_states = np.array(self.tron.get_input_info())
            past_states = np.array(self.tron.get_memorize_board_info())
            if self.tron.check_turn() == self.tron.one_client_koma:

                # すでにゲームが終わっていないか確認する
                if self.tron.can_move():  # 自分自身が動けるか
                    action = self.brain.get_action(now_states)  # 次の一手を決定する

                    can_put = False
                    while not can_put:
                        if self.debug: print("one action:", action)
                        # 駒が置けたならば
                        if self.tron.put_one_koma(direction=action):
                            now_states = np.array(self.tron.get_input_info())
                            past_states = np.array(self.tron.get_memorize_board_info())
                            if self.debug:
                                self.tron.print_board()  # ボード表示
                                print("正しい場所に駒を置けました。")
                            # if self.debug: print("この状態から")
                            # self.tron.print_board(board=past_states, is_debug=True)
                            # if self.debug: print("この状態に変化")
                            # self.tron.print_board(board=now_states, is_debug=True)
                            # if self.debug: print("価値は0")
                            self.brain.train(past_states, now_states, action, reward=0, is_finished=False)
                            self.tron.change_turn()  # 順番を交代
                            can_put = True  # 駒が置けました
                        else:  # 駒が置けないなら
                            if self.debug: print("正しい場所に駒を置けませんでした。")
                            # if self.debug: print("この状態での")
                            # if self.debug: print("このアクションが原因です:", action)
                            self.brain.train(now_states, past_states, action, reward=-100, is_finished=True)  # 第二引数意味なし
                            self.reset()  # 試合終了ボード初期化
                            done = True  # 処理終了
                            can_put = True
                            continue

                else:  # False: 動けない
                    if self.debug: print("駒が動けませんでした。")
                    # print("この状態での")
                    # self.tron.print_board(board=past_states, is_debug=True)
                    # print("この行動が原因でした", self.tron.memory.get_memory_two_action())
                    self.brain.train(past_states, past_states, self.tron.memory.get_memory_one_action(), reward=-100, is_finished=True)  # 第二引数意味なし
                    self.reset()  # 試合終了ボード初期化
                    done = True  # 処理終了
                    continue
            else:
                if self.tron.can_move():
                    action = -1

                    while not self.tron.put_two_koma(action):
                        action = randint(0, 3)

                    if self.debug:
                        print("two action:", action)
                        self.tron.print_board()

                    self.tron.change_turn()  # 順番を交代
                else:
                    if self.debug: print("two loss")
                    self.brain.train(past_states, now_states, self.tron.memory.get_memory_one_action(), reward=100, is_finished=True)
                    self.reset()
                    done = True  # 処理終了
                    self.win += 1
                    continue
