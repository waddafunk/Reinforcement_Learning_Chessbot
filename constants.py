# Number of transitions sampled from the replay buffer
BATCH_SIZE = 128
# Discount factor as mentioned in the previous section
GAMMA = 0.99
# Starting value of epsilon
EPS_START = 0.9
# Final value of epsilon
EPS_END = 0.05
# Controls the rate of exponential decay of epsilon, higher means a slower decay
EPS_DECAY = 2
# Update rate of the target network
TAU = 0.005
# Learning rate
LR = 1e-4

N_ACTIONS = 64 * 63 + 8 * 8 * 4
