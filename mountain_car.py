"""
The implementation of the cartpole game 
"""

from game import game

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('MountainCar-v0')
env.seed(0)

max_t = 200
gamma = 0.95
print_every = 100


class Policy(nn.Module):
    # neural network for the policy
    # TODO: change NN architecture to the optimal for the mountain-car game
    def __init__(self, state_size=2, action_size=3, hidden_size=32):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        # we just consider 1 dimensional probability of action
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action)


class mountain_car(game):
    def __init__(self):
        # NN that represents the policy we optimize over
        self.policy = Policy().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)

    def generate_data(self, estimator, number_of_runs=5, number_of_episodes=2000):
        """
        generate csv table consisting of 3d tuples (return, number of episodes, CI)
        """
        # TODO: add generation of the CSV table

        n_episodes = number_of_episodes
        scores_deque = deque(maxlen=100)
        scores = []
        for e in range(1, n_episodes):
            saved_log_probs = []
            rewards = []
            state = env.reset()

            epsilon = max(1 - n_episodes/(n_episodes*0.8), 0.01)

            # Collect trajectory
            for t in range(max_t):

                action, log_prob = self.policy.act(state)
                if np.random.rand() < epsilon:
                    action = np.random.randint(3)

                # Sample the action from current policy
                saved_log_probs.append(log_prob)
                next_state, reward, done, _ = env.step(action)
                if next_state[0] - state[0] > 0 and action == 2:
                    reward = 1
                if next_state[0] - state[0] < 0 and action == 0:
                    reward = 1

                rewards.append(reward)
                state = next_state.copy()
                if done:
                    break
            # Calculate total expected reward
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))

            trajectory = {'probs': saved_log_probs, 'rewards': rewards}
            self.optimizer_step(estimator, trajectory)
            print(np.mean(scores_deque))
            if np.mean(scores_deque) >= 195.0:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    e - 100, np.mean(scores_deque)))
                break
        return scores

    def optimizer_step(self, estimator, trajectory):
        """
        computes the policy loss  
        """

        # computes policy loss for one trajectory
        policy_loss = estimator.compute_loss(trajectory, gamma)

        # backprop
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
