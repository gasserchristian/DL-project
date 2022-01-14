"""
The implementation of the pendulum
"""

from game import game

import abc

import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

from policies import GaussianPolicy



print_every = 1
store_every=10

class pendulum(game):
    def __init__(self):
        super(pendulum, self).__init__()

        self.gamma = 0.98
        self.env = gym.make('Pendulum-v1')
        self.reset()

    def reset(self, seed=42):
        self.reset_seeds(seed)

        self.snapshot_policy = GaussianPolicy(input_dim=3, output_dim=1) # policy "snapshot" network used by some algorithms
        self.policy = GaussianPolicy(input_dim=3, output_dim=1) # policy network parameters
        self.sample_policy = GaussianPolicy(input_dim=3, output_dim=1) # sample policy used during evaluation

    def sample(self, max_t = 1000, eval = 0):
        """
        sample a trajectory
        {state, action, log_prob, reward}
        snaphsot = True iff we sample from snapshot policy
        snapshot = False iff we sample from current policy
        max_t - maximum length of the trajectory
        """

        # If in evaluation mode, random sample
        if eval:
            policy = self.sample_policy
        else:
            policy = self.policy
        states = []
        actions = []
        saved_log_probs = []
        rewards = []
        state = self.env.reset()
        # Collect trajectory
        for t in range(max_t):
            states.append(state)
            action, log_prob = policy.act(state)
            actions.append(action)
            saved_log_probs.append(log_prob)
            next_state, reward, done, _ = self.env.step(action)
            reward = reward.numpy()[0]

            state = next_state.copy()
            # self.env.render()
            rewards.append(reward) # or after break? reward of terminal state?
            if done:
                print(f"Done {t}")

                break
        trajectory = {'states': states, 'actions': actions,
                        'probs': saved_log_probs, 'rewards': rewards}

        if self.number_of_sampled_trajectories % print_every == 0:
            print(sum(rewards))

        if self.number_of_sampled_trajectories % store_every == 0:
            self.rewards_buffer.append(sum(rewards))
        self.number_of_sampled_trajectories += 1
        return trajectory
