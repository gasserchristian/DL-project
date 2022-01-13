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
store_every = 10
class lunar_lander(game):
    def __init__(self):
        super(lunar_lander, self).__init__()

        self.gamma = 0.98
        self.env = gym.make('LunarLander-v2')
        self.reset()

    def reset(self, seed=42):
        self.reset_seeds(seed)


        self.snapshot_policy = Basic_Policy(state_size=8, action_size=2, hidden_size=32) # policy "snapshot" network used by some algorithms
        self.policy = Basic_Policy(state_size=8, action_size=2, hidden_size=32) # policy network parameters
        self.sample_policy = Basic_Policy(state_size=8, action_size=2, hidden_size=32) # sample policy used during evaluation

  
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
            state, reward, done, _ = self.env.step(action)
            rewards.append(reward) # or after break? reward of terminal state?
            if done:
                print(f"DONE {t} {reward}")
                break
        trajectory = {'states': states, 'actions': actions,
                        'probs': saved_log_probs, 'rewards': rewards}

        if self.number_of_sampled_trajectories % print_every == 0:
            print(sum(rewards))
        if self.number_of_sampled_trajectories % store_every == 0:
            self.rewards_buffer.append(sum(rewards))
        self.number_of_sampled_trajectories += 1
        return trajectory

    