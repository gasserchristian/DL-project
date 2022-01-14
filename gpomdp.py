from estimator import Estimator
import torch
import torch.optim as optim
import statistics
import numpy as np
from lunar_lander import lunar_lander
from cart_pole import cart_pole


class Gpomdp(Estimator):
    def __init__(self, game, args):
        # default lr = 5e-2
        self.optimizer = optim.Adam(game.policy.parameters(), lr=args["lr"])
        self.B = args["mini_batch_size"]  # batch size


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
            policy_param.grad = total_gradient[policy_name]

        self.optimizer.step()  # optimizer step


    def gpomdp_gradient_estimate(self, trajectory, game):
        """
        computes GPOMDP gradient estimate using some trajectory
        """
    
        policy_network = game.policy
        gamma = game.gamma  # discount factor

        trajectory = trajectory.copy()

        log_probs = trajectory['probs']
        rewards = trajectory['rewards']

        policy_loss = -log_probs*rewards  
        policy_loss = policy_loss.sum()
            
        policy_network.zero_grad()
        policy_loss.backward()

        grad_dict = {k: v.grad for k, v in policy_network.named_parameters()}  # one alternative way to extract gradients
        return grad_dict  # returns dictionary!



