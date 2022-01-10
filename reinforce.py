from estimator import Estimator
import torch
import torch.optim as optim


class Reinforce(Estimator):
    def __init__(self, game, B=100):
        self.optimizer = optim.Adam(game.policy.parameters(), lr=1e-2)
        self.B = B  # batch size

    def step(self, game):
        for i in range(self.B):
            trajectory = game.sample()
            gradient_estimator = self.reinforce_gradient_estimate(trajectory,
                                                                  game)  # compute gradient estimate for Reinforce
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

    def reinforce_gradient_estimate(self, trajectory, game):
        """
                computes Reinforce gradient estimate using given trajectory
                """
        policy_network = game.policy
        gamma = game.gamma  # discount factor

        trajectory = trajectory.copy()

        log_probs = trajectory['probs']
        rewards = trajectory['rewards']

        # this nested function computes total discounted reward of the episode
        def discounted_total_reward(rewards):
            discounts = [gamma ** i for i in range(len(rewards) + 1)]
            R = sum([a * b for a, b in zip(discounts, rewards)])
            return R

        total_rewards = discounted_total_reward(rewards)

        policy_loss = []

        for log_prob in log_probs:
            policy_loss.append(log_prob * total_rewards)

            # After that, we concatenate whole policy loss in 0th dimension
            policy_loss = torch.cat(policy_loss).sum()

            policy_network.zero_grad()
            policy_loss.backward()

            grad_dict = {k: v.grad for k, v in
                         policy_network.named_parameters()}  # one alternative way to compute gradient

            return grad_dict  # returns dictionary!
