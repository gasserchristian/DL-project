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


from policies import Basic_Policy




print_every = 100
store_every=10

class mountain_car(game):
    def __init__(self):
        super(mountain_car, self).__init__()

        self.gamma = 0.95

        self.env = gym.make('MountainCar-v0').unwrapped
        self.reset()

    def reset(self, seed=42):
        
        self.reset_seeds(seed)

        self.snapshot_policy = Basic_Policy(state_size=2, action_size=3, hidden_size=16)
        self.policy = Basic_Policy(state_size=2, action_size=3, hidden_size=16)
        self.sample_policy = Basic_Policy(state_size=2, action_size=3, hidden_size=16)

    def sample(self, max_t = 1000, eval = 0):
        """
        sample a trajectory
        {state, action, log_prob, reward}
        snaphsot = True iff we sample from snapshot policy
        snapshot = False iff we sample from current policy
        max_t - maximum length of the trajectory
        """

        # If in evaluation mode, random sample
        
        states = []
        actions = []
        saved_log_probs = []
        rewards = []
        state = self.env.reset()
        # Collect trajectory
        for t in range(max_t):
            states.append(state)
            action, log_prob = self.policy.act(state)
            actions.append(action)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = self.env.step(action)
            rewards.append(reward) # or after break? reward of terminal state?
            if done:
                print(f"DONE {t}")
                break
        trajectory = {'states': states, 'actions': actions,
                        'probs': saved_log_probs, 'rewards': rewards}

        if self.number_of_sampled_trajectories % print_every == 0:
            print(sum(rewards))
        if self.number_of_sampled_trajectories % store_every == 0:
            self.rewards_buffer.append(sum(rewards))
        self.number_of_sampled_trajectories += 1
        return trajectory

  