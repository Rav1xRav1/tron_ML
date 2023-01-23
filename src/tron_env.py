from random import randint
import numpy as np


class Tron_Env:
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

        self.blank: int = 0  # 空白のマス
        self.one_client_koma: int = 1  # クライアント1の駒
        self.two_client_koma: int = -1  # クライアント2の駒
        self.obstacle: int = 2  # 障害物 テストコードでは使用しない

        self.order_of_koma = self.one_client_koma  # 駒の順番を記憶する

        # 以前置いていた場所を記憶する
        self.before_action_one_koma = 0
        self.before_action_two_koma = 0

        self.game_board: list = [self.blank] * (self.MAX_LINE * self.MAX_COLUMN)  # ゲームボード情報をクリア

        # 各コマをセットする
        self.game_board[self.one_posi] = self.one_client_koma
        self.game_board[self.two_posi] = self.two_client_koma

        # 障害物セット
        # self.game_board[self.one_posi+1] = self.obstacle

        # 駒の８個前までの情報を記憶する
        self.memorize_one_client: list = [0] * 8
        self.memorize_two_client: list = [0] * 8

        # スタート位置の記憶
        self.list_rotate_and_add(client_num=self.one_client_koma, new_value=self.one_start_posi)
        self.list_rotate_and_add(client_num=self.two_client_koma, new_value=self.two_start_posi)

        self.memorize_board_info: list = self.get_input_info()  # ひとつ前のボード情報をリストとして保持

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

        # 今誰の盤なのかを記憶
        self.order_of_koma = self.one_client_koma

        # 以前置いていた場所を記憶する
        self.before_action_one_koma = 0
        self.before_action_two_koma = 0

        # 駒記憶状況をリセット
        self.memorize_one_client: list = [0] * 8
        self.memorize_two_client: list = [0] * 8

        # 各クライアントの位置情報を更新する
        self.list_rotate_and_add(client_num=self.one_client_koma, new_value=self.one_start_posi)
        self.list_rotate_and_add(client_num=self.two_client_koma, new_value=self.two_start_posi)

        self.memorize_board_info: list = self.get_input_info()  # ひとつ前のボード情報を保持

    def list_rotate_and_add(self, client_num: int, new_value: int) -> None:

        """
        リスト要素を左に一つ寄せて、新要素を追加する
        :param client_num: クライアントの番号 1: one or -1: two
        :param new_value: 新しく追加する値
        :return: 新しく作られたリスト
        """
        if client_num == self.one_client_koma:
            self.memorize_one_client = self.memorize_one_client[1:] + [new_value]
        else:
            self.memorize_two_client = self.memorize_two_client[1:] + [new_value]

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
        elif (now_posi + self.MAX_COLUMN) < (self.MAX_COLUMN * self.MAX_LINE) and self.game_board[
            now_posi + self.MAX_COLUMN] == self.blank:
            return True
        elif (now_posi % self.MAX_COLUMN) != 0 and self.game_board[now_posi - 1] == self.blank:
            return True
        elif ((now_posi + 1) % self.MAX_COLUMN) != 0 and self.game_board[now_posi + 1] == self.blank:
            return True
        else:
            # 移動不可
            return False

    def can_put(self, posi: int, action: int) -> (bool, int):
        """
        駒が該当場所におけるか確認
        :param posi: 現在の駒の位置
        :param action: 進行方向
        :return:
        """
        if not self.game_board[posi] != 0:
            return False, 0

        # 各方向に対して検査
        if action == 0 and ((posi + 1) % self.MAX_COLUMN) != 0 and self.game_board[posi + 1] == self.blank:
            return True, posi + 1
        elif action == 1 and (posi + self.MAX_COLUMN) < (self.MAX_COLUMN * self.MAX_LINE) and self.game_board[
            posi + self.MAX_COLUMN] == self.blank:
            return True, posi + self.MAX_COLUMN
        elif action == 2 and (posi % self.MAX_COLUMN) != 0 and self.game_board[posi - 1] == self.blank:
            return True, posi - 1
        elif action == 3 and (posi - self.MAX_COLUMN) >= 0 and self.game_board[posi - self.MAX_COLUMN] == self.blank:
            return True, posi - self.MAX_COLUMN
        else:
            return False, 0

    def print_board(self) -> None:
        """
        ボードを表示
        :return: None
        """
        for index in range(self.MAX_COLUMN * self.MAX_LINE):
            print(self.game_board[index], end="")
            if index % self.MAX_COLUMN != self.MAX_COLUMN - 1:
                print(" | ", end="")
            else:
                print()

    def put_one_koma(self, action) -> bool:
        """
        クライアント１の駒を置く
        :param action: 進行方向
        :return: 駒が置けた: True | 置けなかった: False
        """
        can, posi = self.can_put(self.one_posi, action)  # 該当場所に駒が置けるか
        if can:  # 置けるなら
            self.memorize_board_info = self.get_input_info()  # ボード情報の記憶
            self.game_board[posi] = self.one_client_koma  # ボード更新
            self.before_action_one_koma = action  # 以前置いていた場所として記憶
            self.one_posi = posi  # ポジション記憶

            self.list_rotate_and_add(client_num=self.one_client_koma, new_value=posi)  # 行動を記録
            # print("記録しました:", self.memorize_one_client)
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
            self.memorize_board_info = self.get_input_info()  # ボード情報記憶
            self.game_board[posi] = self.two_client_koma  # ボード更新
            self.before_action_two_koma = action  # 以前置いていた場所として記憶
            self.two_posi = posi  # ポジション記憶

            self.list_rotate_and_add(client_num=self.two_client_koma, new_value=posi)  # 行動を記録
            return True  # 試合続行
        else:  # 置けないなら
            return False  # 負け

    def get_board_info(self) -> list:
        """
        ゲームボードの情報を返す
        :return: ボード情報
        """
        return self.game_board.copy()

    def __conbert_1d_to_2d(self, l: list, cols: int = 5) -> list:
        """
        入ってきた一次元配列を二次元配列にして返す
        :param get_list: 一次元配列
        :return: 二次元配列
        """
        return [l[i:i+cols] for i in range(0, len(l), cols)]

    def __get_client_one_info(self) -> list:
        """
        クライアント1のみのリストを返す
        :return: クライアント1
        """
        l = [1 if x == self.one_client_koma else 0 for x in self.get_board_info()]
        return self.__conbert_1d_to_2d(l)

    def __get_client_two_info(self) -> list:
        """
        クライアント2のみのリストを返す
        :return: クライアント2
        """
        l = [1 if x == self.two_client_koma else 0 for x in self.get_board_info()]
        return self.__conbert_1d_to_2d(l)

    def __get_obstacle_info(self) -> list:
        """
        障害物のみのリストを返す
        :return: 障害物
        """
        l = [1 if x == self.obstacle else 0 for x in self.get_board_info()]
        return self.__conbert_1d_to_2d(l)

    def __get_client_one_posi_info(self) -> list:
        """
        クライアント1の現在の場所のみを示すリストを返す
        :return: クライアント1の現在位置
        """
        l = ([0] * (self.MAX_COLUMN * self.MAX_LINE))
        l[self.one_posi] = 1
        return self.__conbert_1d_to_2d(l)

    def __get_client_two_posi_info(self) -> list:
        """
        クライアント2の現在の場所のみを示すリストを返す
        :return: クライアント2の現在位置
        """
        l = ([0] * (self.MAX_COLUMN * self.MAX_LINE))
        l[self.two_posi] = 1
        return self.__conbert_1d_to_2d(l)

    def get_input_info(self) -> list:
        return [self.__get_client_one_info(),
                self.__get_client_two_info(),
                self.__get_obstacle_info(),
                self.__get_client_one_posi_info(),
                self.__get_client_two_posi_info()]

    def get_memorize_board_info(self) -> list:
        """
        ひとつ前のボード情報を返す
        :return: ボード情報
        """
        return self.memorize_board_info.copy()

    def get_before_action(self, client: int) -> int:
        if client == self.one_client_koma:
            return self.before_action_one_koma
        else:
            return self.before_action_two_koma


def main():
    tron = Tron_Env(max_line=5, max_column=5, one_start_posi=11, two_start_posi=13)
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
                # print(tron.get_input_info())
                # print(tron.get_memorize_board_info())
                tron.print_board()  # ボード表示
        else:
            tron.print_board()
            print("one loss")
            tron.board_reset()
            continue

        tron.change_turn()  # 選手交代

        if tron.can_move():
            put = randint(0, 4)  # 置く駒を乱数で決定する
            while not tron.put_two_koma(put):  # 駒が置けるまで再試行する
                put = randint(0, 4)

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
