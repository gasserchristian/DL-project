from estimator import estimator
import torch

class reinforce(estimator):
	def compute_loss(self, trajectory, gamma):
		
		log_probs = trajectory['probs']
		rewards = trajectory['rewards']

		# this nested function computes total discounted reward of the episode 
		def discounted_total_reward(rewards): 
			discounts = [gamma ** i for i in range(len(rewards) + 1)]
			R = sum([a * b for a,b in zip(discounts, rewards)])
			return R
	
		R = discounted_total_reward(rewards)
		policy_loss = []

		for log_prob in log_probs:
			# Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
			policy_loss.append(-log_prob * R)
		# After that, we concatenate whole policy loss in 0th dimension
		policy_loss = torch.cat(policy_loss).sum()
		return policy_loss