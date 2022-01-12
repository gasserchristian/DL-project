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



class GaussianPolicy(nn.Module):
    def __init__(self,
                 input_dim =2,
                 output_dim = 1,
                 hidden_nonlinearity=nn.Tanh()):
        super().__init__()

        self.model = nn.Sequential(*[
            nn.Linear(input_dim, 16),
            hidden_nonlinearity,
            nn.Linear(16, 8),
            hidden_nonlinearity,
            nn.Linear(8, output_dim)
        ]).to(device)

        self.variance = torch.eye(output_dim, device=device) * 1e-3
        
    def forward(self, x):
        
        return torch.distributions.multivariate_normal.MultivariateNormal(
            self.model(x), covariance_matrix=self.variance)

    def act(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # probs = self.forward(state).cpu()
        # model = Categorical(probs)
        # action = model.sample()
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        dist = self.forward(state)
        

        # print(action)
        sample = dist.sample()
        sample = torch.clip(sample, -1, 1)
        log_prob = dist.log_prob(sample)

        return sample.cpu(), log_prob.cpu()

    def log_prob(self, state, action): # probability of taking action "action" in state "state"
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        log_prob = probs.log_prob(action)
        return log_prob.cpu()
