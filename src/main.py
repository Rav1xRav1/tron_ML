import numpy as np
import datetime as dt
from random import randint

import torch
# import torch_directml
import torch.nn as nn
from torch import optim
from torch import multiprocessing as mp
from torch.multiprocessing import Value, Array

from tron import Tron
from model import Model
from brain import Brain
from agent import Agent


def play_do(agent: Agent, process_number: int):
    for i in range(100000):
        """
        if (i + 1) % 10 == 0:
            print("game:", agent.game_count)
            print("win:", agent.win_count)
            print("loss:", agent.loss_count)
            print("miss:", agent.miss_action, end="\n")
            agent.counter_reset()
        """

        agent.play(debug=False)

        if i % 1000 == 0:
            # print(agent.brain.net.c)
            print(f"play_num: {i} | finished time: {dt.datetime.now()} | chosen action: {torch.argmax(agent.brain.net(torch.tensor(agent.env.get_input_info()).float()))} | can putted: {agent.can_putted}")
            agent.counter_reset()


def main():
    start_time = dt.datetime.now()
    print("処理開始:", start_time)

    """
    # テンソルをdirectMLデバイスに送信するためのラッパー
    if torch_directml.is_available():
        device = "privateuseone"
    else:
        device = "cpu"
    """

    model = Model()
    # model.share_memory()

    is_gpu = False  # GPUを使用するか否か

    # model = torch.load("D:/Python/machine_learning/src/tron_dqn.pth")
    brain = Brain(model, is_gpu, debug=False, device="cpu")

    max_column = 1
    max_line = 10

    print("クライアント１のスタート位置:", int(max_line / 2 * max_column + max_column / 4))
    tron = Tron(max_line=max_line, max_column=max_column,
                one_start_posi=int(max_line / 2 * max_column + max_column / 4),
                two_start_posi=int((max_line / 2 + 1) * max_column - (max_column / 4 + 1)))

    agent = Agent(brain, env=tron)

    # print(torch.argmax(model(torch.tensor(tron.get_input_info()).float())))

    play_do(agent=agent, process_number=0)
    exit()

    num_processes = 3
    processes = []

    for rank in range(num_processes):
        p = mp.Process(target=play_do, args=(agent, rank))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    finish_time = dt.datetime.now()
    print("処理終了:", finish_time)
    print("実行時間:", finish_time - start_time)

    torch.save(brain.net, f="tron_dqn.pth")


if __name__ == "__main__":
    main()
