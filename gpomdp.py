from estimator import Estimator
import torch
import torch.optim as optim
import statistics


# average of rewards-to-go serves as baseline
class Gpomdp(Estimator):
    def __init__(self, game, B=10):
        self.optimizer = optim.Adam(game.policy.parameters(), lr=1e-2)
        self.B = B  # batch size

        #set sample policy to current policy
        game.sample_policy = game.policy

    def step(self, game):
        for i in range(self.B):
            trajectory = game.sample()
            gradient_estimator = self.gpomdp_gradient_estimate(trajectory, game)  # compute gradient estimate for GPOMDP
            if i == 0:
                gradient_estimators = gradient_estimator
                continue
            gradient_estimators = self.sum_dictionaries(gradient_estimators, gradient_estimator)

        # compute the total gradient from all trajectories
        total_gradient = {k: v / self.B for k, v in gradient_estimators.items()}

        policy_network = game.policy


        # Here we manually set gradients of all network parameters
        for (policy_name, policy_param) in policy_network.named_parameters():
            policy_param.grad = -total_gradient[policy_name]

        self.optimizer.step()  # optimizer step


    def gpomdp_gradient_estimate(self, trajectory, game):
        """
        computes GPOMDP gradient estimate using given trajectory
        """
        policy_network = game.policy
        gamma = game.gamma  # discount factor

        trajectory = trajectory.copy()

        log_probs = trajectory['probs']
        rewards = trajectory['rewards']

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

        policy_network.zero_grad()
        policy_loss.backward()

        grad_dict = {k: v.grad for k, v in
                     policy_network.named_parameters()}  # one alternative way to compute gradient

        return grad_dict  # returns dictionary!
