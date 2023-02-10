class Memory:
    """
    駒やボードの情報を保持しておくクラス
    """
    def __init__(self):
        self.__memorize_state: list = []
        self.__now_state: list = []

        self.__memorize_one_position: int = 0
        self.__memorize_two_position: int = 0

        self.__memorize_one_action: int = 0
        self.__memorize_two_action: int = 0

    def add_now_state(self, new_state: list):
        self.__add_memory_state(self.get_now_state())
        self.__now_state = new_state.copy()

    def __add_memory_state(self, new_state: list):
        self.__memorize_state = new_state.copy()

    def add_memory_one_position(self, new_position):
        self.__memorize_one_position = new_position

    def add_memory_two_position(self, new_position):
        self.__memorize_two_position = new_position

    def add_memory_one_action(self, new_action):
        self.__memorize_one_action = new_action

    def add_memory_two_action(self, new_action):
        self.__memorize_two_action = new_action

    def get_now_state(self) -> list:
        return self.__now_state.copy()

    def get_memory_state(self) -> list:
        return self.__memorize_state.copy()

    def get_memory_one_position(self) -> int:
        return self.__memorize_one_position

    def get_memory_two_position(self) -> int:
        return self.__memorize_two_position

    def get_memory_one_action(self) -> int:
        return self.__memorize_one_action

    def get_memory_two_action(self) -> int:
        return self.__memorize_two_action
