from random import randint, random
import itertools
import numpy as np

from memory import Memory


class Tron:
    def __init__(self, max_line, max_column, one_start_posi, two_start_posi):

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
        self.one_client_koma: int = 5  # クライアント1の駒
        self.one_client_top_koma: int = 10  # クライアント1の頭
        self.two_client_koma: int = -5  # クライアント2の駒
        self.two_client_top_koma: int = -10  # クライアント2の駒
        self.obstacle: int = -3  # 障害物

        # 情報保持用クラス
        self.memory = Memory(one=self.one_client_koma, one_top=self.one_client_top_koma,
                             two=self.two_client_koma, two_top=self.two_client_top_koma,
                             obstacle=self.obstacle, board_height=self.MAX_LINE, board_wight=self.MAX_COLUMN)

        self.order_of_koma = self.one_client_koma  # 駒の順番を記憶する

        self.game_board: list = [self.blank] * (self.MAX_LINE * self.MAX_COLUMN)  # ゲームボード情報をクリア

        # 各コマをセットする
        self.game_board[self.one_start_posi] = self.one_client_top_koma
        self.game_board[self.two_start_posi] = self.two_client_top_koma

        # 障害物セット
        self.game_board = self.set_obstacle(self.game_board)

        # 状態保存
        self.memory.add_now_state(self.game_board)

    def board_reset(self) -> None:
        """
        ゲームボードの情報をリセットする
        :return: None
        """
        self.game_board: list = [self.blank] * (self.MAX_LINE * self.MAX_COLUMN)  # ゲームボード情報をクリア

        # 位置を更新
        self.one_posi = self.one_start_posi
        self.two_posi = self.two_start_posi

        # 開始位置に配置
        self.game_board[self.one_start_posi] = self.one_client_top_koma
        self.game_board[self.two_start_posi] = self.two_client_top_koma

        # 障害物を設置
        self.game_board = self.set_obstacle(self.game_board)

        # 今誰の盤なのかを記憶
        self.order_of_koma = self.one_client_koma

        # 状態保存
        self.memory.add_now_state(self.game_board)

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
            return self.order_of_koma
        else:
            self.order_of_koma = self.one_client_koma
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

    def can_put(self, pre_cell: int, direction: int) -> tuple[bool, int]:
        """
        駒が該当場所におけるか確認
        :param pre_cell: 現在の駒の位置
        :param direction: 進行方向
        :return:
        """
        # 方向が一致 and その方向に移動可能 and (0以上orボードサイズ未満) and 移動先が空白
        if direction == 0 and (pre_cell + 1) % self.MAX_COLUMN != 0 and pre_cell + 1 <= self.BOARD_SIZE and self.game_board[pre_cell + 1] == self.blank:
            return True, pre_cell + 1
        elif direction == 1 and pre_cell + self.MAX_COLUMN < self.BOARD_SIZE and self.game_board[pre_cell + self.MAX_COLUMN] == self.blank:
            return True, pre_cell + self.MAX_COLUMN
        elif direction == 2 and pre_cell % self.MAX_COLUMN != 0 and pre_cell - 1 >= 0 and self.game_board[pre_cell - 1] == self.blank:
            return True, pre_cell - 1
        elif direction == 3 and pre_cell - self.MAX_COLUMN >= 0 and self.game_board[pre_cell - self.MAX_COLUMN] == self.blank:
            return True, pre_cell - self.MAX_COLUMN

        return False, 0

    def print_board(self, board: np.array = None, is_debug: bool = False) -> None:
        """
        ボードを表示
        :return: None
        """
        if is_debug: print("---デバッグ用の画面です---")
        if board is None:
            board = self.game_board.copy()
        else:
            # ndarrayを一次元の配列に入れ替える
            board = list(itertools.chain.from_iterable(board.tolist()[0]))
        for index in range(self.MAX_COLUMN * self.MAX_LINE):
            print(f"{board[index]:> 3}|", end="")
            if not index % self.MAX_COLUMN != self.MAX_COLUMN - 1:
                print()

    def put_one_koma(self, direction) -> bool:
        """
        クライアント１の駒を置く
        :param direction: 進行方向
        :return: 駒が置けた: True | 置けなかった: False
        """
        can, posi = self.can_put(self.one_posi, direction)  # 該当方向に駒が置けるか
        if can:  # 置けるなら
            self.game_board[self.one_posi] = self.one_client_koma  # トップの情報を削除
            self.game_board[posi] = self.one_client_top_koma  # ボード更新
            self.memory.add_now_state(new_state=self.game_board)  # ボード情報保存
            self.memory.add_memory_one_action(new_action=direction)  # 以前向いた方向として記憶
            self.one_posi = posi  # ポジション記憶

            return True  # 試合続行
        else:  # 置けないなら
            return False  # 負け

    def put_two_koma(self, direction) -> bool:
        """
        クライアント２の駒を置く
        :param direction: 進行方向
        :return: 駒が置けた: True | 置けなかった: False
        """
        can, posi = self.can_put(self.two_posi, direction)  # 該当方向に駒が置けるか
        if can:  # 置けるなら
            self.game_board[self.two_posi] = self.two_client_koma  # トップ情報を削除
            self.game_board[posi] = self.two_client_top_koma  # ボード更新
            self.memory.add_now_state(new_state=self.game_board)  # ボード情報保存
            self.memory.add_memory_two_action(new_action=direction)  # 以前向いた方向として記憶
            self.two_posi = posi  # ポジション記憶

            return True  # 試合続行
        else:  # 置けないなら
            return False  # 負け

    def get_can_move_direction(self, client_num: int) -> tuple[list, list, list, list]:
        """
        現在の座標から移動可能な方向ならば１不可ならば０で埋めたボードサイズ分の一次元配列を返す
        :param client_num: クライアントの番号
        :return: ０か１で埋められた一次元配列
        """
        # 現在の場所を設定
        if client_num == self.one_client_koma:
            pre_cell = self.one_client_koma
        else:
            pre_cell = self.two_client_koma

        # 右
        return ([1 if self.can_put(direction=0, pre_cell=pre_cell)[0] else 0 for _ in range(self.BOARD_SIZE)],
                [1 if self.can_put(direction=1, pre_cell=pre_cell)[0] else 0 for _ in range(self.BOARD_SIZE)],
                [1 if self.can_put(direction=2, pre_cell=pre_cell)[0] else 0 for _ in range(self.BOARD_SIZE)],
                [1 if self.can_put(direction=3, pre_cell=pre_cell)[0] else 0 for _ in range(self.BOARD_SIZE)])

    def __conbert_1d_to_2d(self, lst: list, cols: int = 0) -> list:
        """
        入ってきた一次元配列を二次元配列にして返す
        :param get_list: 一次元配列
        :return: 二次元配列
        """
        if cols == 0:
            cols = self.MAX_COLUMN
        return [lst[i:i + cols] for i in range(0, len(lst), cols)]

    # この状態に変化
    def get_input_info(self, client_num: int) -> tuple:
        one, two, obstacle = self.memory.get_now_state()
        can_move_direction = self.get_can_move_direction(client_num=client_num)
        return (self.__conbert_1d_to_2d(one[0]),  # クライアント１の状態情報
                self.__conbert_1d_to_2d(one[1]),
                self.__conbert_1d_to_2d(one[2]),
                self.__conbert_1d_to_2d(one[3]),
                self.__conbert_1d_to_2d(one[4]),
                self.__conbert_1d_to_2d(one[5]),
                self.__conbert_1d_to_2d(one[6]),
                self.__conbert_1d_to_2d(one[7]),
                self.__conbert_1d_to_2d(two[0]),  # クライアント２の状態情報
                self.__conbert_1d_to_2d(two[1]),
                self.__conbert_1d_to_2d(two[2]),
                self.__conbert_1d_to_2d(two[3]),
                self.__conbert_1d_to_2d(two[4]),
                self.__conbert_1d_to_2d(two[5]),
                self.__conbert_1d_to_2d(two[6]),
                self.__conbert_1d_to_2d(two[7]),
                self.__conbert_1d_to_2d(obstacle),  # 障害物状態情報
                self.__conbert_1d_to_2d([self.order_of_koma for _ in range(self.BOARD_SIZE)]),  # 今誰の盤なのか
                self.__conbert_1d_to_2d(can_move_direction[0]),  # 祖の駒が移動可能な方向
                self.__conbert_1d_to_2d(can_move_direction[1]),
                self.__conbert_1d_to_2d(can_move_direction[2]),
                self.__conbert_1d_to_2d(can_move_direction[3]))

    # この状態から
    def get_memorize_board_info(self) -> tuple:
        one, two, obstacle = self.memory.get_memory_state()
        return (self.__conbert_1d_to_2d(one),
                self.__conbert_1d_to_2d(two),
                self.__conbert_1d_to_2d(obstacle),
                [self.order_of_koma for _ in range(self.BOARD_SIZE)])

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
                n = i * self.MAX_COLUMN + j
                if board[n] == self.blank: board[n] = self.obstacle
                j = round(random() * self.MAX_COLUMN / 2)
                n = int(i * self.MAX_COLUMN + self.MAX_LINE / 2 + j) - 1
                if board[n] == self.blank: board[n] = self.obstacle

        return board


def main():
    max_line, max_column = 10, 10

    tron = Tron(max_line=max_line, max_column=max_column,
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
