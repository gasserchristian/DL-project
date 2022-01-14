"""
Interface for estimators 
"""

import copy
import statistics
import torch
from abc import abstractmethod, ABCMeta
import numpy as np
from lunar_lander import lunar_lander

class Estimator(metaclass=ABCMeta):

    def sum_dictionaries(self, dic1, dic2):
        """
        This method adds two dictionaries together and returns the resulting dictionary;
        it will be useful for us throughout this module
        """
        sdic = {k: dic1.get(k, 0) + dic2.get(k, 0) for k in dic1.keys()}
        return sdic

    @abstractmethod
    def step(self):
        """
        computes policy loss
        """
        pass


# A common variance-reduced estimator class
class VrEstimator(Estimator):
    def gradient_estimate(self, trajectory, game, snapshot=False):
        """
        computes GPOMDP gradient estimate using given trajectory
        """
        policy_network = game.snapshot_policy if snapshot else game.policy  # we have two policy networks 
        gamma = game.gamma  # discount factor of the game 

        trajectory = trajectory.copy()

        log_probs = [] if snapshot else trajectory['probs']
        rewards = trajectory['rewards']
        states = trajectory['states']
        actions = trajectory['actions']

        if snapshot:  # then we need to recompute logprobs using snapshot network
            log_probs = policy_network.forward(states,actions)[1] # compute log probs using snapshot network
    
        policy_loss = -log_probs*rewards  
        policy_loss = policy_loss.sum()
        policy_network.zero_grad()
        policy_loss.backward()
        
        grad_dict = {k: v.grad for k, v in policy_network.named_parameters()}  # one alternative way to compute gradient
        return grad_dict  # returns dictionary


    def snapshot_update(self, game):
        """ Copies current policy network to snapshot policy network (takes snapshot) """
        
        policy_network = game.policy  # current network
        snapshot_network = game.snapshot_policy  # snapshot network

        policy_dict = policy_network.state_dict()  # current network in dictionary format
        snap_dict = snapshot_network.state_dict()  # snapshot network in dictionary format

        for (policy_name, policy_param), (snap_name, snape_param) in zip(policy_dict.items(), snap_dict.items()):
            snap_dict[snap_name] = policy_dict[policy_name]  # clone current NN to the snapshot NN


    def network_update(self, v, game, first_iteration=False):  # update all weights of current network
        """
        v is a dictionary which contains gradient estimates
        """

        policy_network = game.policy
        policy_dict = policy_network.state_dict()  # network parameters in dictionary format

        policy_dict_before_update = copy.deepcopy(policy_dict)

        # The following two lines are important!
        # Here we manually set gradients of all network parameters
        for (policy_name, policy_param) in policy_network.named_parameters():
            policy_param.grad = v[policy_name]

        if not first_iteration:  # optimizer update for remaining subiterations
            self.optimizer_sub.step()
        else:  # optimizer update for first subiteration
            self.optimizer_first.step()


        policy_dict_after_update = policy_network.state_dict()
        """ calculates magnitude of network parameters update """
        self.calculate_lr(policy_dict_before_update, policy_dict_after_update, first_iteration) 

    def importance_weight(self, trajectory, game):
        """
        This method computes importance weight for the given trajectory
        """

        log_probs = trajectory['probs']
        rewards = trajectory['rewards']
        states = trajectory['states']
        actions = trajectory['actions']

        trajectory_prob_current = []
        trajectory_prob_snapshot = []

        log_prob_current = game.policy.forward(states,actions)[1] # compute log probs using snapshot network
        log_prob_snapshot = game.snapshot_policy.forward(states,actions)[1] # compute log probs using snapshot network
        with torch.no_grad():  # we don't want to track gradient history of weights
            weight = log_prob_snapshot.sum() / log_prob_current.sum()
        return weight


    def outer_loop_estimators(self, game):
        """
        Sample N trajectories and compute GPOMDP gradient estimators
        """
        for i in range(self.N):  # sample self.N trajectories
            trajectory = game.sample()  # trajectory is produced by current policy NN
            gradient_estimator = self.gradient_estimate(trajectory, game)  # compute gradient estimate
            if i == 0:
                gradient_estimators = gradient_estimator
                continue
            gradient_estimators = self.sum_dictionaries(gradient_estimators, gradient_estimator)

        return gradient_estimators


    def inner_loop_estimators(self, game):
        policy_estimates = []
        snap_estimates = []
        weights = []

        # sets alpha value if relevant (e.g. STORM, STORM-PG)
        if hasattr(self, 'alpha'):
            alpha = self.alpha
        else:
            alpha = 0

        for i in range(self.B):  # sample self.B trajectories
            trajectory = game.sample()  # trajectory produced by current policy network
            policy_estimate = self.gradient_estimate(trajectory, game, snapshot=False)  # estimate by current policy
            snap_estimate = self.gradient_estimate(trajectory, game, snapshot=True)  # gradient estimate by snapshot
            weight = self.importance_weight(trajectory, game)  # importance weight

            policy_estimates.append(policy_estimate)
            snap_estimates.append(snap_estimate)
            weights.append(weight)

        sum_weights = sum(weights)
        weights = [weight / sum_weights for weight in weights]  # we normalize weights

        for i in range(len(weights)):
            policy_estimate = policy_estimates[i]
            snap_estimate = snap_estimates[i]
            weight = weights[i]
            to_add = {k: v * (-1) * (1 - alpha) * weight for k, v in snap_estimate.items()}  # Dictionary!
            summ = self.sum_dictionaries(policy_estimate, to_add)
            if i == 0:
                gradient_estimators = summ
                continue
            gradient_estimators = self.sum_dictionaries(gradient_estimators, summ)

        return gradient_estimators

    def calculate_lr(self, old_dict, new_dict, first_iteration):
        """ Here we calculate magnitude of adam update; we need this value to know when to finish subiterations """

        dif_dict = {k: new_dict.get(k, 0) - old_dict.get(k, 0) for k in old_dict.keys()}
        summ = []
        for (name, param) in dif_dict.items():
            squared_sum = torch.sum(torch.square(param))
            summ.append(squared_sum)

        lr = sum(summ)
        if first_iteration:
            self.first_iteration_lr = lr
        else:
            self.main_lr = lr
