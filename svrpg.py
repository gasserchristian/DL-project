from estimator import estimator
import torch
import statistics 
from cart_pole import Policy
import numpy as np
from operator import add

"""
TODO:
1) add importance weights 
2) add no grad to network update? and snapshot update? (important to understand!)
"""

class svrpg(estimator):

	# define here snapshot and current NNs 
	def __init__(self, S = 1000, m = 10, lr = 0.01, N = 200, B = 100):
		self.S = S # number of epochs
		self.m = m # epoch size
		self.N = N # batch size
		self.B = B # mini-batch size
		
		self.s = 0 # counter of epochs 
		self.t = self.m #counter within epoch

		self.mu = None # return of outer loop 

		self.current_policy = Policy() # policy network 
		self.snapshot_policy = Policy() # snapshot neural network 

		self.lr = lr # learning rate
		

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
			trajectory = game.sample() # trajectory produced by current policy network

			gradient_estimator = self.gradient_estimate(trajectory, game, snapshot = False)
			snapshot_estimator = self.gradient_estimate(trajectory, game, snapshot = True)
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
		gamma = game.gamma # discount factor 

		log_probs = [] if snapshot else trajectory['probs']
		rewards = trajectory['rewards']
		states = trajectory['states']
		actions = trajectory['actions']

		if snapshot: # then we need to recompute logprobs using snapshot network 
			while True: 
				state = states.pop(0)
				action = actions.pop(0)
				log_prob = policy_network.log_prob(state,action)
				log_probs.append(log_prob)
				if not states:
					break 


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
			policy_loss.append(log_prob * (gamma ** k) * norm_rewards_to_go[k])
			k += 1

		# After that, we concatenate whole policy loss in 0th dimension
		policy_loss = torch.cat(policy_loss).sum()
		
		policy_network.zero_grad() # otherwise gradients are accumulated! 
		policy_loss.backward()

		gradients = []

		for p in policy_network.parameters():
			gradients.append(p.grad)

		return gradients


class sarahpg(estimator):

	# define here snapshot and current NNs
	def __init__(self, S=1000, m=10, lr=0.01, N=200, B=100):
		self.S = S  # number of epochs
		self.m = m  # epoch size
		self.N = N  # batch size
		self.B = B  # mini-batch size

		self.s = 0  # counter of epochs
		self.t = self.m  # counter within epoch

		self.mu = None  # return of outer loop

		self.current_policy = Policy()  # policy network
		self.snapshot_policy = Policy()  # snapshot neural network

		self.lr = lr  # learning rate

	def importance_weight(self, trajectory, game):
		# TODO: compute importance weight for trajectory between
		# current and old policy network
		return 1

	def step(self, game):  # one step of update
		if self.t == self.m:
			self.outer_loop_update(game)  # outer loop of SARAH-PG algorithm
			self.t = 0  # reset counter within epoch

		self.inner_loop_update(game)  # inner loop of SARAH-PG algorithm
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
			p.data = current_network_weights[k]  # clone current NN to snapshot NN
			k += 1

		####################

		"""
		then we sample a batch of trajectories and compute GPOMDP gradient estimation
		"""

		for i in range(self.N):  # sample a batch of trajectories
			trajectory = game.sample()  # trajectory is produced by current policy NN
			gradient_estimator = self.gradient_estimate(trajectory, game)  # compute gradient estimate
			if i == 0:
				gradient_estimators = gradient_estimator
				continue
			gradient_estimators = list(map(add, gradient_estimators, gradient_estimator))  # and then we sum them up

		self.mu = [x / self.N for x in gradient_estimators]  # and average them out

	def inner_loop_update(self, game):
		gradient_estimators = []
		snapshot_estimators = []
		weights = []

		for i in range(self.B):
			trajectory = game.sample()  # trajectory produced by current policy network

			gradient_estimator = self.gradient_estimate(trajectory, game, snapshot=False)
			snapshot_estimator = self.gradient_estimate(trajectory, game, snapshot=True)

			weight = self.importance_weight(trajectory, game)  # TODO
			gradient_estimator = [x / self.B for x in gradient_estimator]
			to_add = [x / self.B * (-1) * weight + self.mu[i] for i,x in enumerate(snapshot_estimator)]

			if i == 0:
				gradient_estimators = list(map(add, gradient_estimator, to_add))
				continue
			gradient_estimators = list(map(add, gradient_estimators, to_add))

		# Update the stochastic step direction mu recursively
		self.mu = gradient_estimators

		####################
		"""
        here we just update snapshot NN with weights of current NN
        """
		current_network_weights = []
		k = 0

		for p in game.policy.parameters():
			current_network_weights.append(p)

		for p in game.snapshot_policy.parameters():
			p.data = current_network_weights[k]  # clone current NN to snapshot NN
			k += 1

		####################

		self.network_update(gradient_estimators, game)  # then we update current policy network

	def network_update(self, v, game):  # update all weights of policy network
		k = 0
		for p in game.policy.parameters():
			p.data += v[k] * self.lr
			k += 1


	def gradient_estimate(self, trajectory, game, snapshot=False):
		# computes GPOMDP gradient estimate using some trajectory
		policy_network = game.snapshot_policy if snapshot else game.policy  # don't forget, we have two networks
		gamma = game.gamma  # discount factor

		log_probs = [] if snapshot else trajectory['probs']
		rewards = trajectory['rewards']
		states = trajectory['states']
		actions = trajectory['actions']

		if snapshot:  # then we need to recompute logprobs using snapshot network
			while True:
				state = states.pop(0)
				action = actions.pop(0)
				log_prob = policy_network.log_prob(state, action)
				log_probs.append(log_prob)
				if not states:
					break

			# this nested function computes a list of rewards-to-go

		def rewards_to_go(rewards):
			rewards_to_go = []
			for i in range(len(rewards) + 1):
				discounts = [gamma ** i for i in range(len(rewards) + 1 - i)]
				reward_to_go = sum([a * b for a, b in zip(discounts, rewards[i::])])
				rewards_to_go.append(reward_to_go)
			return rewards_to_go

		rewards_to_go = rewards_to_go(rewards)
		mean_over_returns = statistics.mean(rewards_to_go)
		norm_rewards_to_go = [reward_to_go - mean_over_returns for reward_to_go in rewards_to_go]

		policy_loss = []
		k = 0  # counter

		for log_prob in log_probs:
			policy_loss.append(log_prob * (gamma ** k) * norm_rewards_to_go[k])
			k += 1

		# After that, we concatenate whole policy loss in 0th dimension
		policy_loss = torch.cat(policy_loss).sum()

		policy_network.zero_grad()  # otherwise gradients are accumulated!
		policy_loss.backward()

		gradients = []

		for p in policy_network.parameters():
			gradients.append(p.grad)

		return gradients


class stormpg (estimator):

	# define here snapshot and current NNs
	def __init__(self, S=1000, m=10, lr=0.01, alpha = 0.5, N=200, B=100):
		self.S = S  # number of epochs
		self.m = m  # epoch size
		self.N = N  # batch size
		self.B = B  # mini-batch size
		self.alpha = alpha  # weighing STORM hyperparameter in the range [0,1]

		self.s = 0  # counter of epochs
		self.t = self.m  # counter within epoch

		self.mu = None  # return of outer loop

		self.current_policy = Policy()  # policy network
		self.snapshot_policy = Policy()  # snapshot neural network

		self.lr = lr  # learning rate

	def importance_weight(self, trajectory, game):
		# TODO: compute importance weight for trajectory between
		# current and old policy network
		return 1

	def step(self, game):  # one step of update
		if self.t == self.m:
			self.outer_loop_update(game)  # outer loop of STORM-PG algorithm
			self.t = 0  # reset counter within epoch

		self.inner_loop_update(game)  # inner loop of STORM-PG algorithm
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
			p.data = current_network_weights[k]  # clone current NN to snapshot NN
			k += 1

		####################

		"""
		then we sample a batch of trajectories and compute GPOMDP gradient estimation
		"""

		for i in range(self.N):  # sample a batch of trajectories
			trajectory = game.sample()  # trajectory is produced by current policy NN
			gradient_estimator = self.gradient_estimate(trajectory, game)  # compute gradient estimate
			if i == 0:
				gradient_estimators = gradient_estimator
				continue
			gradient_estimators = list(map(add, gradient_estimators, gradient_estimator))  # and then we sum them up

		self.mu = [x / self.N for x in gradient_estimators]  # and average them out

	def inner_loop_update(self, game):
		gradient_estimators = []
		snapshot_estimators = []
		weights = []

		for i in range(self.B):
			trajectory = game.sample()  # trajectory produced by current policy network

			gradient_estimator = self.gradient_estimate(trajectory, game, snapshot=False)
			snapshot_estimator = self.gradient_estimate(trajectory, game, snapshot=True)

			weight = self.importance_weight(trajectory, game)  # TODO
			gradient_estimator = [x / self.B for x in gradient_estimator]
			to_add = [(1-self.alpha) * (x / self.B * (-1) * weight + self.mu[i]) for i,x in enumerate(snapshot_estimator)]

			if i == 0:
				gradient_estimators = list(map(add, gradient_estimator, to_add))
				continue
			gradient_estimators = list(map(add, gradient_estimators, to_add))

		# Update the stochastic step direction mu recursively
		self.mu = gradient_estimators

		####################
		"""
        here we just update snapshot NN with weights of current NN
        """
		current_network_weights = []
		k = 0

		for p in game.policy.parameters():
			current_network_weights.append(p)

		for p in game.snapshot_policy.parameters():
			p.data = current_network_weights[k]  # clone current NN to snapshot NN
			k += 1

		####################

		self.network_update(gradient_estimators, game)  # then we update current policy network

	def network_update(self, v, game):  # update all weights of policy network
		k = 0
		for p in game.policy.parameters():
			p.data += v[k] * self.lr
			k += 1


	def gradient_estimate(self, trajectory, game, snapshot=False):
		# computes GPOMDP gradient estimate using some trajectory
		policy_network = game.snapshot_policy if snapshot else game.policy  # don't forget, we have two networks
		gamma = game.gamma  # discount factor

		log_probs = [] if snapshot else trajectory['probs']
		rewards = trajectory['rewards']
		states = trajectory['states']
		actions = trajectory['actions']

		if snapshot:  # then we need to recompute logprobs using snapshot network
			while True:
				state = states.pop(0)
				action = actions.pop(0)
				log_prob = policy_network.log_prob(state, action)
				log_probs.append(log_prob)
				if not states:
					break

			# this nested function computes a list of rewards-to-go

		def rewards_to_go(rewards):
			rewards_to_go = []
			for i in range(len(rewards) + 1):
				discounts = [gamma ** i for i in range(len(rewards) + 1 - i)]
				reward_to_go = sum([a * b for a, b in zip(discounts, rewards[i::])])
				rewards_to_go.append(reward_to_go)
			return rewards_to_go

		rewards_to_go = rewards_to_go(rewards)
		mean_over_returns = statistics.mean(rewards_to_go)
		norm_rewards_to_go = [reward_to_go - mean_over_returns for reward_to_go in rewards_to_go]

		policy_loss = []
		k = 0  # counter

		for log_prob in log_probs:
			policy_loss.append(log_prob * (gamma ** k) * norm_rewards_to_go[k])
			k += 1

		# After that, we concatenate whole policy loss in 0th dimension
		policy_loss = torch.cat(policy_loss).sum()

		policy_network.zero_grad()  # otherwise gradients are accumulated!
		policy_loss.backward()

		gradients = []

		for p in policy_network.parameters():
			gradients.append(p.grad)

		return gradients

class pagepg (estimator):

	# define here snapshot and current NNs
	def __init__(self, S=1000, m=10, lr=0.01, prob = None, N=200, B=100):
		self.S = S  # number of epochs
		self.m = m  # epoch size
		self.N = N  # batch size
		self.B = B  # mini-batch size
		if prob is None:
			self.prob = self.B/(self.N + self.B) # switching probability

		self.s = 0  # counter of epochs
		self.t = self.m  # counter within epochs
		self.p = 1 # if 1, compute full gradient calculation; if 0, do SARAH
		self.mu = None  # return of outer loop

		self.current_policy = Policy()  # policy network
		self.snapshot_policy = Policy()  # snapshot neural network

		self.lr = lr  # learning rate

	def importance_weight(self, trajectory, game):
		# TODO: compute importance weight for trajectory between
		# current and old policy network
		return 1

	def step(self, game):  # one step of update
		if self.p:
			self.full_grad_update(game)  # full grad calculation of PAGE-PG algorithm
		else:
			self.sarah_inner_update(game)  # inner loop of PAGE-PG algorithm
		self.t += 1 # update counter for step updates
		self.p = np.random.choice(2,p = [1 - self.prob, self.prob])
	def full_grad_update(self, game):

		####################
		"""
		here we just update snapshot NN with weights of current NN
		"""
		current_network_weights = []
		k = 0

		for p in game.policy.parameters():
			current_network_weights.append(p)

		for p in game.snapshot_policy.parameters():
			p.data = current_network_weights[k]  # clone current NN to snapshot NN
			k += 1

		####################

		"""
		then we sample a batch of trajectories and compute GPOMDP gradient estimation
		"""

		for i in range(self.N):  # sample a batch of trajectories
			trajectory = game.sample()  # trajectory is produced by current policy NN
			gradient_estimator = self.gradient_estimate(trajectory, game)  # compute gradient estimate
			if i == 0:
				gradient_estimators = gradient_estimator
				continue
			gradient_estimators = list(map(add, gradient_estimators, gradient_estimator))  # and then we sum them up

		self.mu = [x / self.N for x in gradient_estimators]  # and average them out
		self.network_update(self.mu, game)  # then we update current policy network

	def sarah_inner_update(self, game):
		gradient_estimators = []
		snapshot_estimators = []
		weights = []

		for i in range(self.B):
			trajectory = game.sample()  # trajectory produced by current policy network

			gradient_estimator = self.gradient_estimate(trajectory, game, snapshot=False)
			snapshot_estimator = self.gradient_estimate(trajectory, game, snapshot=True)

			weight = self.importance_weight(trajectory, game)  # TODO
			gradient_estimator = [x / self.B for x in gradient_estimator]
			to_add = [ x / self.B * (-1) * weight + self.mu[i] for i,x in enumerate(snapshot_estimator)]

			if i == 0:
				gradient_estimators = list(map(add, gradient_estimator, to_add))
				continue
			gradient_estimators = list(map(add, gradient_estimators, to_add))

		# Update the stochastic step direction mu recursively
		self.mu = gradient_estimators

		####################
		"""
        here we just update snapshot NN with weights of current NN
        """
		current_network_weights = []
		k = 0

		for p in game.policy.parameters():
			current_network_weights.append(p)

		for p in game.snapshot_policy.parameters():
			p.data = current_network_weights[k]  # clone current NN to snapshot NN
			k += 1

		####################

		self.network_update(gradient_estimators, game)  # then we update current policy network

	def network_update(self, v, game):  # update all weights of policy network
		k = 0
		for p in game.policy.parameters():
			p.data += v[k] * self.lr
			k += 1


	def gradient_estimate(self, trajectory, game, snapshot=False):
		# computes GPOMDP gradient estimate using some trajectory
		policy_network = game.snapshot_policy if snapshot else game.policy  # don't forget, we have two networks
		gamma = game.gamma  # discount factor

		log_probs = [] if snapshot else trajectory['probs']
		rewards = trajectory['rewards']
		states = trajectory['states']
		actions = trajectory['actions']

		if snapshot:  # then we need to recompute logprobs using snapshot network
			while True:
				state = states.pop(0)
				action = actions.pop(0)
				log_prob = policy_network.log_prob(state, action)
				log_probs.append(log_prob)
				if not states:
					break

			# this nested function computes a list of rewards-to-go

		def rewards_to_go(rewards):
			rewards_to_go = []
			for i in range(len(rewards) + 1):
				discounts = [gamma ** i for i in range(len(rewards) + 1 - i)]
				reward_to_go = sum([a * b for a, b in zip(discounts, rewards[i::])])
				rewards_to_go.append(reward_to_go)
			return rewards_to_go

		rewards_to_go = rewards_to_go(rewards)
		mean_over_returns = statistics.mean(rewards_to_go)
		norm_rewards_to_go = [reward_to_go - mean_over_returns for reward_to_go in rewards_to_go]

		policy_loss = []
		k = 0  # counter

		for log_prob in log_probs:
			policy_loss.append(log_prob * (gamma ** k) * norm_rewards_to_go[k])
			k += 1

		# After that, we concatenate whole policy loss in 0th dimension
		policy_loss = torch.cat(policy_loss).sum()

		policy_network.zero_grad()  # otherwise gradients are accumulated!
		policy_loss.backward()

		gradients = []

		for p in policy_network.parameters():
			gradients.append(p.grad)

		return gradients


class page_stormpg (estimator):

	# define here snapshot and current NNs
	def __init__(self, S=1000, m=10, lr=0.01, prob = None, alpha = 0.5, N=200, B=100):
		self.S = S  # number of epochs
		self.m = m  # epoch size
		self.N = N  # batch size
		self.B = B  # mini-batch size
		if prob is None:
			self.prob = self.B/(self.N + self.B) # switching probability

		self.s = 0  # counter of epochs
		self.t = self.m  # counter within epochs
		self.alpha = alpha
		self.p = 1 # if 1, compute full gradient calculation; if 0, do SARAH
		self.mu = None  # return of outer loop

		self.current_policy = Policy()  # policy network
		self.snapshot_policy = Policy()  # snapshot neural network

		self.lr = lr  # learning rate

	def importance_weight(self, trajectory, game):
		# TODO: compute importance weight for trajectory between
		# current and old policy network
		return 1

	def step(self, game):  # one step of update
		if self.p:
			self.full_grad_update(game)  # full grad calculation of PAGE-PG algorithm
		else:
			self.storm_inner_update(game)  # inner loop of PAGE-PG algorithm
		self.t += 1 # update counter for step updates
		self.p = np.random.choice(2,p = [1 - self.prob, self.prob])
	def full_grad_update(self, game):

		####################
		"""
		here we just update snapshot NN with weights of current NN
		"""
		current_network_weights = []
		k = 0

		for p in game.policy.parameters():
			current_network_weights.append(p)

		for p in game.snapshot_policy.parameters():
			p.data = current_network_weights[k]  # clone current NN to snapshot NN
			k += 1

		####################

		"""
		then we sample a batch of trajectories and compute GPOMDP gradient estimation
		"""

		for i in range(self.N):  # sample a batch of trajectories
			trajectory = game.sample()  # trajectory is produced by current policy NN
			gradient_estimator = self.gradient_estimate(trajectory, game)  # compute gradient estimate
			if i == 0:
				gradient_estimators = gradient_estimator
				continue
			gradient_estimators = list(map(add, gradient_estimators, gradient_estimator))  # and then we sum them up

		self.mu = [x / self.N for x in gradient_estimators]  # and average them out
		self.network_update(self.mu, game)  # then we update current policy network

	def storm_inner_update(self, game):
		gradient_estimators = []
		snapshot_estimators = []
		weights = []

		for i in range(self.B):
			trajectory = game.sample()  # trajectory produced by current policy network

			gradient_estimator = self.gradient_estimate(trajectory, game, snapshot=False)
			snapshot_estimator = self.gradient_estimate(trajectory, game, snapshot=True)

			weight = self.importance_weight(trajectory, game)  # TODO
			gradient_estimator = [x / self.B for x in gradient_estimator]
			to_add = [(1-self.alpha) * (x / self.B * (-1) * weight + self.mu[i]) for i,x in enumerate(snapshot_estimator)]

			if i == 0:
				gradient_estimators = list(map(add, gradient_estimator, to_add))
				continue
			gradient_estimators = list(map(add, gradient_estimators, to_add))

		# Update the stochastic step direction mu recursively
		self.mu = gradient_estimators

		####################
		"""
        here we just update snapshot NN with weights of current NN
        """
		current_network_weights = []
		k = 0

		for p in game.policy.parameters():
			current_network_weights.append(p)

		for p in game.snapshot_policy.parameters():
			p.data = current_network_weights[k]  # clone current NN to snapshot NN
			k += 1

		####################

		self.network_update(gradient_estimators, game)  # then we update current policy network

	def network_update(self, v, game):  # update all weights of policy network
		k = 0
		for p in game.policy.parameters():
			p.data += v[k] * self.lr
			k += 1


	def gradient_estimate(self, trajectory, game, snapshot=False):
		# computes GPOMDP gradient estimate using some trajectory
		policy_network = game.snapshot_policy if snapshot else game.policy  # don't forget, we have two networks
		gamma = game.gamma  # discount factor

		log_probs = [] if snapshot else trajectory['probs']
		rewards = trajectory['rewards']
		states = trajectory['states']
		actions = trajectory['actions']

		if snapshot:  # then we need to recompute logprobs using snapshot network
			while True:
				state = states.pop(0)
				action = actions.pop(0)
				log_prob = policy_network.log_prob(state, action)
				log_probs.append(log_prob)
				if not states:
					break

			# this nested function computes a list of rewards-to-go

		def rewards_to_go(rewards):
			rewards_to_go = []
			for i in range(len(rewards) + 1):
				discounts = [gamma ** i for i in range(len(rewards) + 1 - i)]
				reward_to_go = sum([a * b for a, b in zip(discounts, rewards[i::])])
				rewards_to_go.append(reward_to_go)
			return rewards_to_go

		rewards_to_go = rewards_to_go(rewards)
		mean_over_returns = statistics.mean(rewards_to_go)
		norm_rewards_to_go = [reward_to_go - mean_over_returns for reward_to_go in rewards_to_go]

		policy_loss = []
		k = 0  # counter

		for log_prob in log_probs:
			policy_loss.append(log_prob * (gamma ** k) * norm_rewards_to_go[k])
			k += 1

		# After that, we concatenate whole policy loss in 0th dimension
		policy_loss = torch.cat(policy_loss).sum()

		policy_network.zero_grad()  # otherwise gradients are accumulated!
		policy_loss.backward()

		gradients = []

		for p in policy_network.parameters():
			gradients.append(p.grad)

		return gradients
