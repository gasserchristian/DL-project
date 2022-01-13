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
import random
from torch.distributions import Categorical
from collections import deque
from policies import Basic_Policy

import os
import time


print_every=100
store_every=10

class cart_pole(game):
	def __init__(self):

		super(cart_pole, self).__init__()
		self.gamma = 1.0

		self.env = gym.make('CartPole-v0')
	
		self.reset()

	def reset(self, seed=42):
		# TODO: perform reset of policy networks
		self.reset_seeds(seed)
		

		self.snapshot_policy = Basic_Policy() # policy "snapshot" network used by some algorithms
		self.policy = Basic_Policy() # policy network parameters
		self.sample_policy = Basic_Policy() # sample policy used during evaluation

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
				break
		trajectory = {'states': states, 'actions': actions,
						'probs': saved_log_probs, 'rewards': rewards}

		if self.number_of_sampled_trajectories % print_every == 0:
			print(sum(rewards))
		if self.number_of_sampled_trajectories % store_every == 0:
			self.rewards_buffer.append(sum(rewards))
		self.number_of_sampled_trajectories += 1
		return trajectory

	