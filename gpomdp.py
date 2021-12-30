from estimator import estimator
import torch
import statistics 

# average of rewards-to-go serves as baseline 
class gpomdp(estimator):
	def compute_loss(self, trajectory, gamma):
		
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
		return policy_loss