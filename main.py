import gc
import math
import random
from itertools import count

import matplotlib.pyplot as plt
import torch
from chess import BLACK, Move
from torch import nn, optim

from constants import (BATCH_SIZE, EPS_DECAY, EPS_END, EPS_START, GAMMA, LR,
                       N_ACTIONS, TAU)
from deep_q_learning import DQN, ReplayMemory, Transition
from environment import ChessEnv

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

env = ChessEnv()


state = env.reset()
N_OBSERVATIONS = 64

policy_net = DQN(N_OBSERVATIONS, N_ACTIONS, abs((N_OBSERVATIONS - N_ACTIONS) // 20)).to(
    device
)
target_net = DQN(N_OBSERVATIONS, N_ACTIONS, abs((N_OBSERVATIONS - N_ACTIONS) // 20)).to(
    "cpu"
)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state, turn, net="policy"):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if net == "policy":
        state.cuda()
        network = policy_net
    else:
        state = state.cpu()
        network = target_net

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = network(state).max(1).indices.view(1, 1).cuda()
            if turn == BLACK:
                from_square, to_square, promotion_map = env.action_to_move(action)
                from_square = 9 - from_square
                to_square = 9 - to_square
                action = torch.tensor(
                    [
                        [
                            env.move_to_action(
                                Move(from_square, to_square, promotion=promotion_map)
                            )
                        ]
                    ],
                    device=device,
                    dtype=torch.int32,
                )
            if net == "target":
                state.cuda()
            return action
    else:
        return torch.tensor([[env.sample()]], device=device, dtype=torch.int32)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device="cpu")
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states.cpu()).max(1).values
        )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.cpu()

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(
        state_action_values.cpu(), expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    print("episode:", i_episode)

    # Initialize the environment and get its state
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        torch.cuda.empty_cache()
        gc.collect()
        action = select_action(state, env.turn())
        observation, reward, terminated, truncated = env.step(action.item())
        print("Action:", action.item(), env.explain_action(action.item()))
        print("episode:", i_episode, "t:", t, "reward:", reward)
        env.render()
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            if env.turn() == BLACK:
                observation = -observation
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, torch.tensor(action, device=device), next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.cpu().state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)
        policy_net.cuda()

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print("Complete")
plot_durations(show_result=True)
plt.ioff()
plt.show()
