from estimator import estimator
import torch

class reinforce(estimator):
	def compute_loss(self, trajectory):
		saved_log_probs = trajectory[0]
		R = trajectory[1][0]

		policy_loss = []

		for log_prob in saved_log_probs:
			# Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
			policy_loss.append(-log_prob * R)
		# After that, we concatenate whole policy loss in 0th dimension
		policy_loss = torch.cat(policy_loss).sum()
		return policy_loss