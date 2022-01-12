"""
This module contains parametrizations for policies
""" 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Basic_Policy(nn.Module):
	""" 
	our simplest policy network
	we use it for discrete cart pole game
	"""

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
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		probs = self.forward(state).cpu()
		model = Categorical(probs)
		action = model.sample()
		return action.item(), model.log_prob(action)

	def log_prob(self, state, action): # returns the probability of taking given action in given state 
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		probs = self.forward(state).cpu()
		model = Categorical(probs)
		action = torch.tensor([action])
		log_prob = model.log_prob(action)
		return log_prob

