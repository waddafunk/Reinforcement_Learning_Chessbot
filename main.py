import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from environment import ChessEnv


# Define the RL agent
class ChessAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma  # discount factor
        self.Q = self._build_model(state_dim, action_dim)
        self.Q.compile(optimizer="adam", loss="mse")
        self.Q_target = self.Q.copy()

    def _build_model(self, state_dim, action_dim, shrink=3):
        # Build a neural network model for the agent
        model = Sequential(
            [Dense(state_dim, activation="relu", input_shape=(state_dim,))]
            + list(
                reversed(
                    [
                        Dense(dimension, activation="relu")
                        for dimension in range(action_dim, state_dim, shrink)
                    ]
                )
            )
            + [
                Dense(
                    action_dim, activation="linear"
                ),  # keep last activation linear for numerical stability during training
            ]
        )
        return model

    def get_action(self, state):
        state = state.flatten()
        raw_output = self.model.predict(state[np.newaxis, ...])[0][0]

        # Transform the output to the desired range
        scaled_output = (tf.sigmoid(raw_output) * 20480).numpy()
        action = int(np.round(scaled_output))

        # Ensure the output is within the valid range
        return max(0, min(action, 20480))

    def train(self, state, action, reward, next_state, done):
        # Convert inputs to appropriate shape
        state = np.array(state).flatten().reshape(1, self.state_dim)
        next_state = np.array(next_state).reshape(1, self.state_dim)

        # Get the current Q values for the state
        current_q = self.model.predict(state)

        # Get the Q values for the next state
        next_q = self.model.predict(next_state)

        # Update the Q value for the action taken
        if done:
            current_q[0][action] = reward
        else:
            current_q[0][action] = reward + self.gamma * np.max(next_q)

        # Train the model
        self.model.fit(state, current_q, verbose=0)

    def update_epsilon(self, episode, total_episodes):
        # Implement epsilon decay for exploration-exploitation trade-off
        self.epsilon = max(0.01, 1.0 - (episode / total_episodes))


# Main training loop
def train():
    env = ChessEnv()
    agent = ChessAgent(64, 1)

    for episode in range(10):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state

        # Evaluate the agent's performance periodically
        # if episode % eval_interval == 0:
        #    evaluate(agent)


# Evaluation function
def evaluate(agent):
    # Evaluate the agent's performance against a baseline or human player
    pass


# Run the training
train()
