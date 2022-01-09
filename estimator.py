from abc import abstractmethod, ABCMeta
import statistics
import torch
from operator import add


class Estimator(metaclass=ABCMeta):

    @abstractmethod
    def step(self):
        """
		computes policy loss
		"""
        pass

# A common variance-reduced estimator class
class VrEstimator(Estimator):
    def gradient_estimate(self, trajectory, game, snapshot=False):
        # computes GPOMDP gradient estimate using some trajectory
        policy_network = game.snapshot_policy if snapshot else game.policy  # don't forget, we have two networks
        gamma = game.gamma  # discount factor

        log_probs = [] if snapshot else trajectory['probs']
        rewards = trajectory['rewards']
        states = trajectory['states']
        actions = trajectory['actions']

        if snapshot:  # then we need to recompute logprobs using snapshot network
            while True:
                state = states.pop(0)
                action = actions.pop(0)
                log_prob = policy_network.log_prob(state, action)
                log_probs.append(log_prob)
                if not states:
                    break

        # this nested function computes a list of rewards-to-go

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

        policy_network.zero_grad()  # otherwise gradients are accumulated!
        policy_loss.backward()

        gradients = []

        for p in policy_network.parameters():
            gradients.append(p.grad)

        return gradients

    def network_update(self, v, game):  # update all weights of policy network
        k = 0
        for p in game.policy.parameters():
            p.data += v[k] * self.lr
            k += 1

    def importance_weight(self, trajectory, game):
        # TODO: compute importance weight for trajectory between
        # current and old policy network
        return 1

    def snapshot_update(self, game):
        ####################
        """
        here we just update snapshot NN with weights of current NN
        """
        current_network_weights = []
        k = 0

        for p in game.policy.parameters():
            current_network_weights.append(p)

        for p in game.snapshot_policy.parameters():
            p.data = current_network_weights[k]  # clone current NN to snapshot NN
            k += 1

    def gpomd_estimators(self, N, game):
        """
        Sample N trajectories and compute GPOMDP gradient estimators
        """
        for i in range(N):  # sample a batch of trajectories
            trajectory = game.sample()  # trajectory is produced by current policy NN
            gradient_estimator = self.gradient_estimate(trajectory, game)  # compute gradient estimate
            if i == 0:
                gradient_estimators = gradient_estimator
                continue
            gradient_estimators = list(map(add, gradient_estimators, gradient_estimator))  # and then we sum them up

        return gradient_estimators
