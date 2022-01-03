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

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v0')
env.seed(0)

print_every=100

class Policy(nn.Module):
	# neural network for the policy
	# TODO: change NN architecture to the optimal for the cart-pole game
	def __init__(self, state_size=4, action_size=2, hidden_size=32):
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

	def log_prob(self, state, action): # probability of taking action "action" in state "state"
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		probs = self.forward(state).cpu()
		model = Categorical(probs)
		action = torch.tensor([action])
		log_prob = model.log_prob(action)
		return log_prob



class cart_pole(game):
	def __init__(self):
		self.gamma = 1.0
		self.number_of_sampled_trajectories = 0 # total number of sampled trajectories

		self.snapshot_policy = Policy().to(device) # policy "snapshot" network used by some algorithms
		self.policy = Policy().to(device) # policy network

		# self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)

	def reset(self):
		# TODO: perform reset of policy networks
		pass


	def sample(self, max_t = 1000):
		"""
		sample a trajectory
		{state, action, log_prob, reward}
		snaphsot = True iff we sample from snapshot policy
		snapshot = False iff we sample from current policy
		max_t - maximum length of the trajectory
		"""
		states = []
		actions = []
		saved_log_probs = []
		rewards = []
		state = env.reset()
		# Collect trajectory
		for t in range(max_t):
			states.append(state)
			action, log_prob = self.policy.act(state)
			actions.append(action)
			saved_log_probs.append(log_prob)
			state, reward, done, _ = env.step(action)
			rewards.append(reward) # or after break? reward of terminal state?
			if done:
				break
		trajectory = {'states': states, 'actions': actions,
						'probs': saved_log_probs, 'rewards': rewards}

		if self.number_of_sampled_trajectories % 10 == 0:
			print(sum(rewards))
		self.number_of_sampled_trajectories += 1
		return trajectory

	def evaluate(self, number_of_runs = 10): # performs the evaluation of the current policy NN for
											 # a given number of runs
		number_of_sampled_trajectories = self.number_of_sampled_trajectories
		results = [np.sum(self.sample(200)['rewards']) for i in range(number_of_runs)]
		self.number_of_sampled_trajectories = number_of_sampled_trajectories
		
		# TODO:
		# it should return 3 values:
		# 1) self.number_of_sampled_trajectories
		# 2) mean performance
		# 3) confidence interval
		return (self.number_of_sampled_trajectories,np.mean(results),np.std(results))

	def generate_data(self, estimator, number_of_sampled_trajectories = 1000):
		"""
		generate a file of 3d tuples: (number of sample trajectories, mean reward, CI)
		until it reaches the specified number of trajectories ("number_of_sampled_trajectories")
		"""
		# trajectories = []
		# mean_reward = []
		# CI_reward = []
		evaluations = []

		while True:
			estimator.step(self) # performs one step of update for the selected estimator
								   # this can be one or more episodes

			# after policy NN updates, we need to evaluate this updated policy using self.evaluate()
			evaluations.append(self.evaluate())
			# TODO: store the returned values: trajectories, mean_reward, CI_reward in some file
			if self.number_of_sampled_trajectories > number_of_sampled_trajectories:
				print("finish")
				self.number_of_sampled_trajectories = 0
				break
		np.savetxt('data--cartpole_'+type(estimator).__name__+'.txt',np.array(evaluations).transpose())
