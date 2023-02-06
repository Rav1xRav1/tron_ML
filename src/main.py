import numpy as np
import datetime as dt
from random import randint

import torch
# import torch_directml
import torch.nn as nn
from torch import optim

from tron import Tron
from model import Model
from brain import Brain
from agent import Agent


def play_do(agent: Agent):
    for i in range(100000):
        agent.play(debug=False)
        if i % 1000 == 0:
            print(f"finished time: {dt.datetime.now()} | chosen action: {torch.argmax(agent.brain.net(torch.tensor(agent.tron.get_input_info()).float()))} | can putted: {agent.can_putted}")
            agent.counter_reset()


def main():
    start_time = dt.datetime.now()
    print("処理開始:", start_time)

    max_column = 10
    max_line = 10

    model = Model(height=max_line, width=max_column, input_channel=1, output_size=4)

    is_gpu = False  # GPUを使用するか否か

    # model = torch.load("D:/Python/machine_learning/src/tron_dqn.pth")
    brain = Brain(model, is_gpu, debug=False, device="cpu")

    print(f"ボードの大きさ: {max_column}*{max_line}")
    print("クライアント１のスタート位置:", int(max_line / 2 * max_column + max_column / 4))

    tron = Tron(max_line=max_line, max_column=max_column,
                one_start_posi=int(max_line / 2 * max_column + max_column / 4),
                two_start_posi=int((max_line / 2 + 1) * max_column - (max_column / 4 + 1)))

    agent = Agent(brain, env=tron)

    # print(torch.argmax(model(torch.tensor(tron.get_input_info()).float())))

    play_do(agent=agent)

    finish_time = dt.datetime.now()
    print("処理終了:", finish_time)
    print("実行時間:", finish_time - start_time)

    torch.save(brain.net, f="tron_dqn.pth")


if __name__ == "__main__":
    main()
