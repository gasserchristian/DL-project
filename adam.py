import numpy as np 
import torch 

"""
Here we implement Adam optimizer as it is described in Appendix D of SVRPG paper by Papini
"""

class adam():
	def __init__(self, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, alpha = 0.05):
		"""
		initialization of adam metaparameters, they are all taken from Papini paper
		and they are optimal for cart pole game!  
		"""
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon # not mentioned in the paper -> we need to tune it 
		self.alpha = alpha 

		self.k_dict = {} # dictionary of estimated first moments 
		self.mu_dict = {} # dictionary of estimated second moments 

		self.t = 1 # current time step 

	def increment_and_lr(self, gradient_estimates):
		"""
		This method takes dictionary of gradient estimates consisting of tensors and it returns:
		1) increment for network parameters (in dictionary format)
		2) "learning rate" that we use in order to make epoch size adaptive (this is real number)
		"""

		increments_dict = {} # this is 1) 

		increment_sum_of_squares = [] # this we need to compute 2) 
		k_sum_of_squares = [] # this we need to compute 2) 

		if ((bool(self.k_dict) == False) or (bool(self.mu_dict) == False)): 
			"""
			if dictionaries "self.mu_dict" and "self.k_dict" are empty, we insert there (keys, 0) pairs, 
			where keys are names of network layers which we take from "gradient_estimates" dictionary
			"""
			for (gradient_name, gradient_values) in gradient_estimates.items():
				self.mu_dict[gradient_name] = 0 
				self.k_dict[gradient_name] = 0

		# here we iterate over 3 dictionaries: 
		for (ku_name, ku_prev), (mu_name, mu_prev), (gradient_name, gradient_values)\
		 in zip(self.k_dict.items(), self.mu_dict.items(), gradient_estimates.items()):
			
			if ((ku_name != mu_name) or (ku_name != gradient_name)):
				print("something is wrong") # all names should be the same 

			k_t = self.beta_1 * ku_prev + (1 - self.beta_1) * gradient_values # this is 1D tensor 
			mu_t = self.beta_2 * mu_prev + (1 - self.beta_2) * (gradient_values ** 2) # this is 1D tensor 

			k_hat_t = k_t / (1 - self.beta_1 ** self.t) # this is 1D tensor
			mu_hat_t = mu_t / (1 - self.beta_2 ** self.t) # this is 1D tensor
			increment = (self.alpha / (torch.sqrt(mu_hat_t) + self.epsilon)) * k_hat_t # this is 1D tensor
			
			increment = increment.detach() # we don't want to track gradient history of this object
			k_t = k_t.detach() # we don't want to track gradient history of this object

			increments_dict[ku_name] = increment 

			increment_sum_of_squares.append(torch.sum(torch.square(increment))) 
			k_sum_of_squares.append(torch.sum(torch.square(k_t)))

			# These two dictionaries keep past values of k and mu as it is described in paper 
			self.k_dict[ku_name] = k_t 
			self.mu_dict[mu_name] = mu_t


		ks_sum = sum(k_sum_of_squares)
		
		if ks_sum == 0: 
			ks_sum = 1 # we need this to avoid division by zero in the first iteration

		increments_sum = sum(increment_sum_of_squares)
		learning_rate = torch.sqrt(increments_sum / ks_sum)
		self.t += 1 # increment timestep 

		return increments_dict, learning_rate


