import torch

# Number of transitions sampled from the replay buffer
BATCH_SIZE = 128
# Discount factor as mentioned in the previous section
GAMMA = 0.99
# Starting value of epsilon
EPS_START = 0.9
# Final value of epsilon
EPS_END = 0.05
# Controls the rate of exponential decay of epsilon, higher means a slower decay
EPS_DECAY = 5000
# Update rate of the target network
TAU = 0.005
# Learning rate
LR = 1e-4

N_ACTIONS = 64 * 63 + 8 * 8 * 4

START_TURNS = 20
END_TURNS = 200
TURNS_COEF = 1e-1

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
