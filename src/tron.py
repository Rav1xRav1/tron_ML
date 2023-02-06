from random import randint, random
import numpy as np

from memory import Memory


class Tron:
    def __init__(self, max_line, max_column, one_start_posi, two_start_posi):
        # 情報保持用クラス
        self.memory = Memory()

        # 各コマの開始位置を記憶する
        self.one_start_posi: int = one_start_posi
        self.two_start_posi: int = two_start_posi

        # 各クライアントのポジションを記憶する
        self.one_posi: int = one_start_posi
        self.two_posi: int = two_start_posi

        # 盤の大きさ
        self.MAX_LINE = max_line
        self.MAX_COLUMN = max_column
        self.BOARD_SIZE = max_line * max_column

        self.blank: int = 0  # 空白のマス
        self.one_client_koma: int = 1  # クライアント1の駒
        self.two_client_koma: int = -1  # クライアント2の駒
        self.obstacle: int = 2  # 障害物

        self.order_of_koma = self.one_client_koma  # 駒の順番を記憶する

        self.game_board: list = [self.blank] * (self.MAX_LINE * self.MAX_COLUMN)  # ゲームボード情報をクリア

        # 各コマをセットする
        self.game_board[self.one_start_posi] = self.one_client_koma
        self.game_board[self.two_start_posi] = self.two_client_koma

        # 障害物セット
        self.game_board = self.set_obstacle(self.game_board)

    def board_reset(self) -> None:
        """
        ゲームボードの情報をリセットする
        :return: None
        """
        self.game_board: list = [self.blank] * (self.MAX_LINE * self.MAX_COLUMN)  # ゲームボード情報をクリア

        # 障害物
        # self.game_board[self.one_start_posi+1] = self.obstacle

        # 位置を更新
        self.one_posi = self.one_start_posi
        self.two_posi = self.two_start_posi

        # 開始位置に配置
        self.game_board[self.one_posi] = self.one_client_koma
        self.game_board[self.two_posi] = self.two_client_koma

        # 障害物を設置
        self.game_board = self.set_obstacle(self.game_board)

        # 今誰の盤なのかを記憶
        self.order_of_koma = self.one_client_koma

    def check_turn(self) -> int:
        """
        現在誰かの順番を返す
        :return: 順番を返す
        """
        return self.order_of_koma

    def change_turn(self) -> int:
        """
        駒の順番を変更する
        :return: 駒の種類を返す
        """
        if self.check_turn() == self.one_client_koma:
            self.order_of_koma = self.two_client_koma
            # print("1➡2")
            return self.order_of_koma
        else:
            self.order_of_koma = self.one_client_koma
            # print("2➡1")
            return self.order_of_koma

    def can_move(self) -> bool:
        """
        駒が現在地から動かせるか確認するメソッド
        :return: 移動可能: True | 移動不可: False
        """
        if self.check_turn() == self.one_client_koma:
            now_posi = self.one_posi
        else:
            now_posi = self.two_posi

        # 各方向に対して検査
        if (now_posi - self.MAX_COLUMN) >= 0 and self.game_board[now_posi - self.MAX_COLUMN] == self.blank:
            return True
        elif (now_posi + self.MAX_COLUMN) < (self.MAX_COLUMN * self.MAX_LINE) and self.game_board[now_posi + self.MAX_COLUMN] == self.blank:
            return True
        elif (now_posi % self.MAX_COLUMN) != 0 and self.game_board[now_posi - 1] == self.blank:
            return True
        elif ((now_posi + 1) % self.MAX_COLUMN) != 0 and self.game_board[now_posi + 1] == self.blank:
            return True
        else:
            # 移動不可
            return False

    def can_put(self, pre_cell: int, direction: int) -> (bool, int):
        """
        駒が該当場所におけるか確認
        :param pre_cell: 現在の駒の位置
        :param direction: 進行方向
        :return:
        """
        # 方向が一致 and その方向に移動可能 and (0以上orボードサイズ未満) and 移動先が空白
        if direction == 0 and (pre_cell + 1) % self.MAX_COLUMN != 0 and pre_cell + 1 >= self.BOARD_SIZE and self.game_board[pre_cell + 1] == self.blank:
            return True, pre_cell + 1
        elif direction == 1 and pre_cell + self.MAX_COLUMN < self.BOARD_SIZE and self.game_board[pre_cell + self.MAX_COLUMN] == self.blank:
            return True, pre_cell + self.MAX_COLUMN
        elif direction == 2 and pre_cell % self.MAX_COLUMN != 0 and pre_cell - 1 >= 0 and self.game_board[pre_cell - 1] == self.blank:
            return True, pre_cell - 1
        elif direction == 3 and pre_cell - self.MAX_COLUMN >= 0 and self.game_board[pre_cell - self.MAX_COLUMN] == self.blank:
            return True, pre_cell - self.MAX_COLUMN

        return False, 0

    def print_board(self) -> None:
        """
        ボードを表示
        :return: None
        """
        for index in range(self.MAX_COLUMN * self.MAX_LINE):
            print(f"{self.game_board[index]:> 3}|", end="")
            if index % self.MAX_COLUMN != self.MAX_COLUMN - 1:
                # print("|", end="")
                pass
            else:
                print()

        print("one position:", self.one_posi)
        print("two position:", self.two_posi)

    def put_one_koma(self, direction) -> bool:
        """
        クライアント１の駒を置く
        :param direction: 進行方向
        :return: 駒が置けた: True | 置けなかった: False
        """
        can, posi = self.can_put(self.one_posi, direction)  # 該当場所に駒が置けるか
        if can:  # 置けるなら
            self.memory.add_memory_state(new_state=self.get_input_info())  # ボード情報保存
            self.game_board[posi] = self.one_client_koma  # ボード更新
            self.memory.add_memory_one_action(new_action=action)  # 以前置いていた場所として記憶
            self.one_posi = posi  # ポジション記憶

            return True  # 試合続行
        else:  # 置けないなら
            return False  # 負け

    def put_two_koma(self, action) -> bool:
        """
        クライアント２の駒を置く
        :param action: 進行方向
        :return: 駒が置けた: True | 置けなかった: False
        """
        can, posi = self.can_put(self.two_posi, action)  # 該当場所に駒が置けるか
        if can:  # 置けるなら
            self.memory.add_memory_state(new_state=self.get_input_info())  # ボード情報保存
            self.game_board[posi] = self.two_client_koma  # ボード更新
            self.memory.add_memory_two_action(new_action=action)  # 以前置いていた場所として記憶
            self.two_posi = posi  # ポジション記憶

            return True  # 試合続行
        else:  # 置けないなら
            return False  # 負け

    def __get_board_info(self) -> list:
        """
        ゲームボードの情報を返す
        :return: ボード情報
        """
        return self.game_board.copy()

    def __conbert_1d_to_2d(self, lst: list, cols: int = 0) -> list:
        """
        入ってきた一次元配列を二次元配列にして返す
        :param get_list: 一次元配列
        :return: 二次元配列
        """
        if cols == 0:
            cols = self.MAX_COLUMN
        self.print_board()
        return [lst[i:i + cols] for i in range(0, len(lst), cols)]

    def __get_game_board_info(self):
        """
        クライアント１とクライアント２の情報を合わせて返す
        :return: ボード情報
        """
        return self.__conbert_1d_to_2d(self.game_board.copy())

    def get_input_info(self) -> list:
        return self.__get_game_board_info()

    def get_before_action(self, client: int) -> int:
        """
        該当クライアントが以前行った行動を返す
        :param client: クライアント番号
        :return: 以前の行動記録
        """
        if client == self.one_client_koma:
            return self.before_action_one_koma
        else:
            return self.before_action_two_koma

    def set_obstacle(self, board: list) -> list:
        """
        障害物をセットして返す
        :param board: ボード情報
        :return: 障害物セット後のボード情報
        """
        board = board.copy()

        level = 5
        offset = round(random() * level)

        for i in range(self.MAX_LINE):
            if (i + offset) % level == 0:
                j = round(random() * self.MAX_COLUMN / 2)
                board[i * self.MAX_COLUMN + j] = self.obstacle
                j = round(random() * self.MAX_COLUMN / 2)
                board[int(i * self.MAX_COLUMN + self.MAX_LINE / 2 + j)] = self.obstacle

        return board

    def get_enemy_cell(self) -> int:
        """
        敵の位置を返す
        :return: 敵の位置
        """
        return self.two_posi


def main():
    max_line, max_column = 10, 10

    tron = Tron_Env(max_line=max_line, max_column=max_column,
                    one_start_posi=int(max_line / 2 * max_column + max_column / 4),
                    two_start_posi=int((max_line / 2 + 1) * max_column - (max_column / 4 + 1)))
    tron.print_board()

    while True:
        if tron.can_move():  # 自分の駒が置けるのなら
            # 駒が置けた: True | 置けなかった: False
            if not tron.put_one_koma(int(input("one: "))):
                tron.print_board()
                print("one loss")
                tron.board_reset()
                continue
            else:
                tron.print_board()  # ボード表示
        else:
            tron.print_board()
            print("one loss")
            tron.board_reset()
            continue

        tron.change_turn()  # 選手交代

        if tron.can_move():
            put = -1
            while not tron.put_two_koma(put):  # 駒が置けるまで再試行する
                put = randint(0, 4)  # 置く駒を乱数で決定する

            print("two:", put)
            tron.print_board()
        else:
            tron.print_board()
            print("two loss")
            tron.board_reset()
            continue

        tron.change_turn()


if __name__ == "__main__":
    main()
