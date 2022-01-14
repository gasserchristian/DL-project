"""
Interface for games
"""

from abc import abstractmethod, ABCMeta
import torch
import numpy as np
import random
import os
import re

parser = lambda x : re.sub(r'([.? \'!]+) *', r'-', x)

class game(metaclass=ABCMeta):
	def __init__(self):
		self.rewards_buffer = []
		self.number_of_sampled_trajectories = 0 # total number of sampled trajectories


	@abstractmethod
	def reset(self):
		pass


	def reset_seeds(self, seed=42):
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.random.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)
		self.env.seed(seed)

		if hasattr(self.env, 'action_space'):
			self.env.action_space.seed(seed)


	def sample(self, max_t, eval):
		pass


	def generate_data(self, estimator, hyper_parameters, sweep_parameter, number_of_sampled_trajectories = 10, number_of_runs = 1, root_path="./"):
		"""
		trains chosen estimator with selected hyperparameters until it reaches "number_of_sampled_trajectories"
		trajectories; the training process is repeated number_of_runs times;
		at the end, it generatess files with all reward data which we later plot 
		"""

		results = []
		for i in range(number_of_runs):
			self.reset(i) # Each run has different seed, but same across estimators
			estimator_instance = estimator(self, hyper_parameters)
			while True:
				estimator_instance.step(self) # performs one step of update for the selected estimator
									   # this can be one or more episodes

				if self.number_of_sampled_trajectories > number_of_sampled_trajectories:
					print(f'finish run {i+1} of {number_of_runs}, length : {len(self.rewards_buffer)}, ntraj {self.number_of_sampled_trajectories}')
					self.number_of_sampled_trajectories = 0
					results.append(self.rewards_buffer)
					self.rewards_buffer = []
					break

		name = f"{type(self).__name__}_{type(estimator_instance).__name__}__{str(number_of_sampled_trajectories)}__{str(number_of_runs)}_{str(estimator_instance.B)}_{sweep_parameter}"
		minLength = np.min([len(item) for item in results])
		for i in range(len(results)):
			results[i] = results[i][:minLength]
		# store a numpy binary

		name += '.npy'
		file_path = os.path.join(root_path, name)
		# store a numpy binary
		np.save(file_path, np.array(results))

