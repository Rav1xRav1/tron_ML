class Memory:
    """
    駒やボードの情報を保持しておくクラス
    """

    def __init__(self, one, one_top, two, two_top, obstacle, board_height, board_wight):
        """
        :param one: クライアント1の駒番号
        :param one_top: クライアント1の駒頭番号
        :param two: クライアント2の駒番号
        :param two_top: クライアント2の駒頭番号
        :param obstacle: 障害物の番号
        """
        self.one_koma, self.one_koma_top = one, one_top
        self.two_koma, self.two_koma_top = two, two_top
        self.obstacle = obstacle

        self.height, self.wight = board_height, board_wight

        self.__memorize_one_koma: list = [[0] * self.height*self.wight] * 10
        self.__memorize_two_koma: list = [[0] * self.height*self.wight] * 10
        self.__memorize_obstacle: list = []

        self.__memorize_one_action: int = 0
        self.__memorize_two_action: int = 0

    def reset_memory(self) -> None:
        """
        各種方法を初期化
        """
        self.__memorize_one_koma: list = [[0] * self.height * self.wight] * 10
        self.__memorize_two_koma: list = [[0] * self.height * self.wight] * 10
        self.__memorize_obstacle = []

        self.__memorize_one_action = 0
        self.__memorize_two_action = 0

    def add_now_state(self, new_state: list) -> None:
        """
        現在の全ての状態を保存する（1つ）
        :param new_state: 現在の全ての情報
        """
        self.add_now_one_state(new_state=[l if l in (self.one_koma, self.one_koma_top) else 0 for l in new_state])
        self.add_now_two_state(new_state=[l if l in (self.two_koma, self.two_koma_top) else 0 for l in new_state])
        self.add_now_obstacle(new_obstacle_state=[l if l == self.obstacle else 0 for l in new_state])

    def add_now_one_state(self, new_state: list) -> None:
        """
        クライアント1の情報を追加、過去の手を7つまで保存
        :param new_state:
        :return:
        """
        self.__memorize_one_koma = self.__memorize_one_koma[1:] + [new_state]

    def add_now_two_state(self, new_state: list) -> None:
        """
        クライアント2の情報を追加、過去の手を7つまで保存
        :param new_state: クライアント2情報
        """
        self.__memorize_two_koma = self.__memorize_two_koma[1:] + [new_state]

    def add_now_obstacle(self, new_obstacle_state: list) -> None:
        """
        障害物情報を追加
        :param new_obstacle_state: 障害物情報
        """
        self.__memorize_obstacle = new_obstacle_state

    def add_memory_one_action(self, new_action) -> None:
        """
        クライアント1のアクションを記憶する
        :param new_action: 新しいアクション
        """
        self.__memorize_one_action = new_action

    def add_memory_two_action(self, new_action) -> None:
        """
        クライアント2のアクションを記憶する
        :param new_action: 新しいアクション
        :return:
        """
        self.__memorize_two_action = new_action

    def get_now_state(self) -> tuple:
        """
        現在のボードの状態を返す
        :return: 現在のボード
        """
        return (self.__memorize_one_koma[1:],
                self.__memorize_two_koma[1:],
                self.__memorize_obstacle[1:])

    def get_memory_state(self) -> tuple:
        """
        一つ過去のボードの状態を返す
        :return: 過去のボード
        """
        return (self.__memorize_one_koma[:-1],
                self.__memorize_two_koma[:-1],
                self.__memorize_obstacle[:-1])

    def get_memory_one_action(self) -> int:
        """
        クライアント1のアクション情報を返す
        :return: クライアント1のアクション
        """
        return self.__memorize_one_action

    def get_memory_two_action(self) -> int:
        """
        クライアント2のアクション情報を返す
        :return: クライアント2のアクション
        """
        return self.__memorize_two_action
