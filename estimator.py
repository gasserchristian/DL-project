import copy
import statistics
import torch
from abc import abstractmethod, ABCMeta
import numpy as np

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
        policy_network = game.snapshot_policy if snapshot else game.policy  # don't forget, we have two networks
        gamma = game.gamma  # discount factor

        trajectory = trajectory.copy()

        log_probs = [] if snapshot else trajectory['probs']
        rewards = trajectory['rewards']
        states = trajectory['states']
        actions = trajectory['actions']

        k = 0

        if snapshot:  # then we need to recompute logprobs using snapshot network
            while True:
                state = states[k]
                action = actions[k]
                log_prob = policy_network.log_prob(state, action)
                log_probs.append(log_prob)
                k += 1
                if k == len(states):
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

        policy_network.zero_grad()
        policy_loss.backward()

        grad_dict = {k: v.grad for k, v in
                     policy_network.named_parameters()}  # one alternative way to compute gradient

        return grad_dict  # returns dictionary!

    def snapshot_update(self, game):
        """
                First of all, we take snapshot -  clone current network to the snapshot network
                """
        policy_network = game.policy  # current network
        snapshot_network = game.snapshot_policy  # snapshot network

        policy_dict = policy_network.state_dict()  # current network in dictionary format
        snap_dict = snapshot_network.state_dict()  # snapshot network in dictionary format

        for (policy_name, policy_param), (snap_name, snape_param) in zip(policy_dict.items(), snap_dict.items()):
            snap_dict[snap_name] = policy_dict[policy_name]  # clone current NN to the snapshot NN

    def sample_policy_update(self, game):
        """
                First, we append the current network parameter to our policy parameter list
                """
        policy_network = game.policy  # current network
        sample_network = game.sample_policy  # sample policy network

        policy_dict = policy_network.state_dict()  # current network in dictionary format
        sample_dict = sample_network.state_dict() # sample network in dictionary format

        temp_dict = copy.deepcopy(policy_dict)  # temporary network in dictionary format

        # append the current policy dictionary into the list
        self.policy_parameter_list.append(temp_dict)

        #Choose random uniform network policy
        index = np.random.choice(len(self.policy_parameter_list))

        for (policy_name, policy_param), (sample_name, sample_param) in zip(sample_dict.items(), self.policy_parameter_list[index].items()):
            # sample_dict[sample_name] = copy.deepcopy((torch.tensor(self.policy_parameter_list[index][policy_name].detach().numpy())))  # clone current NN to the snapshot NN
            sample_dict[sample_name].copy_(self.policy_parameter_list[index][
                                                                       policy_name].detach()) # clone current NN to the snapshot NN


    def network_update(self, v, game, first_iteration=False):  # update all weights of current network
        """
        v is a dictionary which contains gradient estimates
        """

        policy_network = game.policy
        policy_dict = policy_network.state_dict()  # network in dictionary format

        policy_dict_before_update = copy.deepcopy(policy_dict)

        # The following two lines are important!
        # Here we manually set gradients of all network parameters
        for (policy_name, policy_param) in policy_network.named_parameters():
            policy_param.grad = -v[policy_name]

        if not first_iteration:  # optimizer update for remaining subiterations
            self.optimizer_sub.step()
        else:  # optimizer update for first subiteration
            self.optimizer_first.step()

        policy_dict_after_update = policy_network.state_dict()
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

        k = 0

        while True:
            state = states[k]
            action = actions[k]
            log_prob_current = game.policy.log_prob(state, action)
            log_prob_snapshot = game.snapshot_policy.log_prob(state, action)
            trajectory_prob_current.append(log_prob_current)
            trajectory_prob_snapshot.append(log_prob_snapshot)
            k += 1
            if k == len(states):
                break

        log_prob_snapshot = sum(log_prob_snapshot)
        log_prob_current = sum(log_prob_current)

        with torch.no_grad():  # we don't want to track gradient history of weights
            weight = log_prob_snapshot / log_prob_current

        return weight

    def print_snapshot(self, game, show_gradients=False):
        """
        We use this method for debugging, it is not present in the script
        It outputs parameters and gradients of both networks
        """

        policy_network = game.policy
        policy_dict = policy_network.state_dict()

        snapshot_network = game.snapshot_policy
        snap_dict = snapshot_network.state_dict()

        print("THIS IS SNAPSHOT")
        for (policy_name, policy_param) in snap_dict.items():
            print(policy_name)
            print(policy_param)

        print("THIS IS POLICY")
        for (policy_name, policy_param) in policy_dict.items():
            print(policy_name)
            print(policy_param)

        if show_gradients:
            print("THIS IS SNAPSHOT GRADIENTS")
            grad_dict = {k: v.grad for k, v in snapshot_network.named_parameters()}  # gradients
            for (policy_name, policy_param) in grad_dict.items():
                print(policy_name)
                print(policy_param)

            print("THIS IS POLICY GRADIENTS")
            grad_dict = {k: v.grad for k, v in policy_network.named_parameters()}  # gradients
            for (policy_name, policy_param) in grad_dict.items():
                print(policy_name)
                print(policy_param)

        policy_network = game.policy  # current network
        snapshot_network = game.snapshot_policy  # snapshot network

        policy_dict = policy_network.state_dict()  # current network in dictionary format
        snap_dict = snapshot_network.state_dict()  # snapshot network in dictionary format

        for (policy_name, policy_param), (snap_name, snape_param) in zip(policy_dict.items(), snap_dict.items()):
            snap_dict[snap_name].copy_(policy_dict[policy_name])  # clone current NN to the snapshot NN

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
        """
        Here we calculate magnitude of adam update; we need this value to know when to finish subiterations
        """

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
