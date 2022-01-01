from estimator import estimator
import torch
from torch import nn
from torch import optim
import statistics 
from cart_pole import Policy 
import time
import numpy as np
from operator import add

"""
TODO:
1) add importance weights 
2) add no grad to network update? and snapshot update? (important to understand!)
"""

class svrpg(estimator):

	# define here snapshot and current NNs 
	def __init__(self, S = 1000, m = 10, alpha = 0.01, N = 20, B = 10):
		self.S = S # number of epochs
		self.m = m # epoch size
		self.N = N # batch size
		self.B = B # mini-batch size
		
		self.s = 0 # counter of epochs 
		self.t = self.m # counter within epoch 

		self.mu = None # return of outer loop 

		self.current_policy = Policy() # policy network 
		self.snapshot_policy = Policy() # snapshot neural network 

		self.lr = alpha # learning rate 
		

	def importance_weight(self, trajectory, game): 
		# TODO: compute importance weight for trajectory between 
		# current and old policy network
		return 1

	def step(self, game): # one step of update 
		if self.t == self.m:
			self.outer_loop_update(game) # outer loop of SVRPG algprithm 
			self.t = 0 # reset counter within epoch 
		
		self.inner_loop_update(game) # inner loop of SVRPG algprithm 
		self.t += 1

	def outer_loop_update(self, game):
		
		####################
		"""
		here we just update snapshot NN with weights of current NN 
		"""
		current_network_weights = []
		k = 0

		for p in game.policy.parameters():
			current_network_weights.append(p)

		for p in game.snapshot_policy.parameters():
			p.data = current_network_weights[k] # clone current NN to snapshot NN
			k += 1

		####################



		"""
		then we sample a batch of trajectories and compute GPOMDP gradient estimation
		"""

		for i in range(self.N): # sample a batch of trajectories 
			trajectory = game.sample() # trajectory is produced by current policy NN 
			gradient_estimator = self.gradient_estimate(trajectory, game) # compute gradient estimate 
			if i == 0:
				gradient_estimators = gradient_estimator 
				continue
			gradient_estimators = list(map(add, gradient_estimators, gradient_estimator)) # and then we sum them up

		self.mu = [x / self.N for x in gradient_estimators] # and average them out



	def inner_loop_update(self, game):
		gradient_estimators = []
		snapshot_estimators = []
		weights = [] 
		
		for i in range(self.B):
			trajectory = game.sample(snapshot = False) # produced by current policy netowrk 
			trajectory_snapshot = game.sample(snapshot = True) # produced by snapshot policy network

			gradient_estimator = self.gradient_estimate(trajectory, game, snapshot = False)
			snapshot_estimator = self.gradient_estimate(trajectory_snapshot, game, snapshot = True)
			weight = self.importance_weight(trajectory, game) # TODO 

			to_add = [x*(-1)*weight for x in snapshot_estimator] 

			if i == 0:
				gradient_estimators = list(map(add, gradient_estimator, to_add))
				continue
			gradient_estimators = list(map(add, gradient_estimators, to_add))

		c = [x / self.B for x in gradient_estimators]
		v = list(map(add,c,self.mu))
		self.network_update(v, game) # then we update current policy network


	def network_update(self, v, game): # update all weights of policy network
		k = 0
		for p in game.policy.parameters():
			p.data += v[k] * self.lr
			k += 1


	def gradient_estimate(self, trajectory, game, snapshot = False):
		# computes GPOMDP gradient estimate using some trajectory 
		policy_network = game.snapshot_policy if snapshot else game.policy # don't forget, we have two networks
		gamma = game.gamma

		log_probs = trajectory['probs']
		rewards = trajectory['rewards']

		# this nested function computes a list of rewards-to-go 
		def rewards_to_go(rewards): 
			rewards_to_go = []
			for i in range(len(rewards) + 1):
				discounts = [gamma ** i for i in range(len(rewards) + 1 - i)]
				reward_to_go = sum([a * b for a,b in zip(discounts, rewards[i::])])
				rewards_to_go.append(reward_to_go)
			return rewards_to_go

		rewards_to_go = rewards_to_go(rewards)
		mean_over_returns = statistics.mean(rewards_to_go)
		norm_rewards_to_go = [reward_to_go - mean_over_returns for reward_to_go in rewards_to_go]

		policy_loss = []
		k = 0 # counter

		for log_prob in log_probs:
			policy_loss.append(-log_prob * (gamma ** k) * norm_rewards_to_go[k])
			k += 1

		# After that, we concatenate whole policy loss in 0th dimension
		policy_loss = torch.cat(policy_loss).sum()
		policy_loss.backward()

		gradients = []
		for p in policy_network.parameters():
			gradients.append(p.grad)

		return gradients 
		

