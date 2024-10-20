import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque, OrderedDict
from itertools import count, chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if GPU is to be used
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, shrink=3):
        super(DQN, self).__init__()
        layer_dict = OrderedDict(
            chain.from_iterable(
                [
                    (
                        f"linear_{layer_idx}",
                        nn.Linear(n_neurons, n_neurons - shrink),
                    ),
                    (f"relu_{layer_idx}", nn.ReLU()),
                ]
                for layer_idx, n_neurons in enumerate(
                    range(n_observations, n_actions, -shrink)
                )
                if n_neurons - shrink > n_actions
            )
        )

        # append last layer with correct output shape
        last_linear_key = list(layer_dict)[-2]
        last_layer_number = int(last_linear_key.split("_")[1]) + 1
        last_linear_layer = layer_dict[last_linear_key]
        layer_dict.update(
            {
                f"linear_{last_layer_number}": nn.Linear(
                    last_linear_layer.out_features, n_actions
                )
            }
        )

        self.sequential = nn.Sequential(layer_dict)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.sequential(x)
        return x


if __name__ == "__main__":
    print("CUDA available: ", torch.cuda.is_available())
    input_feats = 20
    testnet = DQN(input_feats, 1, 3)
    print(testnet)
    testnet(torch.rand(input_feats))
