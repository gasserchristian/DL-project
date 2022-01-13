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
		self.policy = Policy().to(device) # policy network parameters
		self.sample_policy = Policy().to(device) # sample policy used during evaluation

	def reset(self):
		global env
		# TODO: perform reset of policy networks
		torch.manual_seed(0)
		env.seed(0)
		env = gym.make('CartPole-v0')
		self.snapshot_policy = Policy().to(device)
		self.policy = Policy().to(device)
		self.sample_policy = Policy().to(device)

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
		state = env.reset()
		# Collect trajectory
		for t in range(max_t):
			states.append(state)
			action, log_prob = policy.act(state)
			actions.append(action)
			saved_log_probs.append(log_prob)
			state, reward, done, _ = env.step(action)
			rewards.append(reward) # or after break? reward of terminal state?
			if done:
				break
		trajectory = {'states': states, 'actions': actions,
						'probs': saved_log_probs, 'rewards': rewards}

		if self.number_of_sampled_trajectories % print_every == 0:
			print(sum(rewards))
		self.number_of_sampled_trajectories += 1
		return trajectory

	def evaluate(self): # performs the evaluation of the current policy NN
		# def evaluate(self, number_of_runs = 30):
		number_of_sampled_trajectories = self.number_of_sampled_trajectories
		results = self.sample(200,eval=1)['rewards']
		# results = [np.sum(self.sample(200, eval = 1)['rewards']) for i in range(number_of_runs)]
		self.number_of_sampled_trajectories = number_of_sampled_trajectories

		# TODO:
		# it should return 3 values:
		# 1) self.number_of_sampled_trajectories
		# 2) mean performance
		# 3) confidence interval
		return np.sum(self.sample(200,eval=1)['rewards'])
		# return (self.number_of_sampled_trajectories,np.mean(results),np.std(results))

	def generate_data(self, estimator, number_of_sampled_trajectories = 10000, number_of_runs = 30):
		"""
		generate a file of 3d tuples: (number of sample trajectories, mean reward, CI)
		until it reaches the specified number of trajectories ("number_of_sampled_trajectories")
		"""
		# trajectories = []
		# mean_reward = []
		# CI_reward = []
		results = []
		for _ in range(number_of_runs):
			self.reset()
			estimator_instance = estimator(self)
			evaluations = []
			while True:
				estimator_instance.step(self) # performs one step of update for the selected estimator
									   # this can be one or more episodes
				# after policy NN updates, we need to evaluate this updated policy using self.evaluate()
				evaluations.append((self.number_of_sampled_trajectories,self.evaluate()))
				# TODO: store the returned values: trajectories, mean_reward, CI_reward in some file
				if self.number_of_sampled_trajectories > number_of_sampled_trajectories:
					# print("finish",`${}`)
					print(f'finish run {_+1} of {number_of_runs}')
					self.number_of_sampled_trajectories = 0
					results.append(evaluations)
					break
		# print(np.array(results).shape)
		# store a numpy binary
		np.save('data-runs--'+type(self).__name__+'_'+type(estimator_instance).__name__+'.npy',np.array(results))
		# np.savetxt('data-runs--'+type(self).__name__+'_'+type(estimator_instance).__name__+'.txt',np.array(results))
		# np.savetxt('data--'+type(self).__name__+'_'+type(estimator_instance).__name__+'.txt',np.array(evaluations).transpose())
