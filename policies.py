"""
This module contains a collection of policy networks 
""" 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Basic_Policy(nn.Module):
	"""Basic policy network for experiments"""

	def __init__(self, state_size=4, action_size=2, hidden_size=32):
		super(Basic_Policy, self).__init__()
		self.fc1 = nn.Linear(state_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, action_size)

	def forward(self, state):
		"""
		returns distribution over actions for the input state 
		"""
		x = F.relu(self.fc1(state))
		x = self.fc2(x)
		return F.softmax(x, dim=1)

	def act(self, state): # returns action and its probability for the input state 
		state = torch.from_numpy(state).float().unsqueeze(0)
		probs = self.forward(state)
		model = Categorical(probs)
		action = model.sample()
		return action.item(), model.log_prob(action)

	def log_prob(self, state, action): # returns the probability of taking given action in given state 
		state = torch.from_numpy(state).float().unsqueeze(0)
		probs = self.forward(state)
		model = Categorical(probs)
		action = torch.tensor([action])
		log_prob = model.log_prob(action)
		return log_prob


def mlp(sizes, activation, output_activation=nn.Identity):
    """The basic multilayer perceptron architecture used."""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class CartPole_Policy(nn.Module):
	"""Policy network for CartPole game"""

	def __init__(self, obs_dim=4, act_dim=2, hidden_sizes=32, activation=nn.ReLU):
		super().__init__()  
		self.logits_net = mlp([obs_dim] + [hidden_sizes] + [act_dim], activation)

	def _distribution(self, obs):
		"""Takes the observation and outputs a distribution over actions."""
		logits = self.logits_net(obs)
		return Categorical(logits=logits)

	def _log_prob_from_distribution(self, pi, act):
		"""
		Takes a distribution and action, then gives the log-probability of the action
		under that distribution.
		"""
		return pi.log_prob(act)

	def forward(self, obs, act=None):
		"""
		Produce action distributions for given observations, and then compute the
		log-likelihood of given actions under those distributions.
		"""
		pi = self._distribution(obs)
		logp_a = None
		if act is not None:
			logp_a = self._log_prob_from_distribution(pi, act)
		return pi, logp_a
	
	def step(self, state):
		"""
		Take a state and return an action, and log-likelihood of chosen action.
		"""
		with torch.no_grad():
			action = self.forward(state)[0].sample() 
			log_prob_action = self.forward(state, action)[1]

		# convert tensors to numerical values 
		action = action.detach().numpy()  
		log_prob_action = log_prob_action.detach().numpy()

		return action, log_prob_action


class Lunar_Policy(nn.Module):
	"""Policy network for Lunar Lander game"""

	def __init__(self, obs_dim=8, act_dim=4, hidden_sizes=(64,64), activation=nn.Tanh):
		super().__init__()  
		self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

	def _distribution(self, obs):
		"""Takes the observation and outputs a distribution over actions."""
		logits = self.logits_net(obs)
		return Categorical(logits=logits)

	def _log_prob_from_distribution(self, pi, act):
		"""
		Takes a distribution and action, then gives the log-probability of the action
		under that distribution.
		"""
		return pi.log_prob(act)

	def forward(self, obs, act=None):
		"""
		Produce action distributions for given observations, and then compute the
		log-likelihood of given actions under those distributions.
		"""
		pi = self._distribution(obs)
		logp_a = None
		if act is not None:
			logp_a = self._log_prob_from_distribution(pi, act)
		return pi, logp_a
	
	def step(self, state):
		"""
		Take a state and return an action, and log-likelihood of chosen action.
		"""
		with torch.no_grad():
			action = self.forward(state)[0].sample() 
			log_prob_action = self.forward(state, action)[1]

		# convert tensors to numerical values 
		action = action.detach().numpy()  
		log_prob_action = log_prob_action.detach().numpy()

		return action, log_prob_action


class GaussianPolicy(nn.Module):
	"""Gaussian policy network, for Mountain Car game"""

	def __init__(self, input_dim =2, output_dim = 1, hidden_nonlinearity=nn.Tanh()): 
		super().__init__()

		self.model = nn.Sequential(*[
			nn.Linear(input_dim, 16),
			hidden_nonlinearity,
			nn.Linear(16, 8),
			hidden_nonlinearity,
			nn.Linear(8, output_dim)
		])

		self.variance = torch.eye(output_dim) * 1e-3
        
	def forward(self, x):
        
		return torch.distributions.multivariate_normal.MultivariateNormal(
			self.model(x), covariance_matrix=self.variance)

	def act(self, state):
        
		state = torch.from_numpy(state).float().unsqueeze(0)
		dist = self.forward(state)
		sample = dist.sample()
		sample = torch.clip(sample, -1, 1)
		log_prob = dist.log_prob(sample)

		return sample, log_prob

	def log_prob(self, state, action): # probability of taking action "action" in state "state"
		state = torch.from_numpy(state).float().unsqueeze(0)
		probs = self.forward(state)
		log_prob = probs.log_prob(action)
		return log_prob
