"""
The implementation of the cartpole game 
"""

from game import game

import abc

import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

from policies import GaussianPolicy

# from garage.torch.modules import GaussianMLPModule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
env = gym.make('MountainCarContinuous-v0')
env.seed(0)

max_t = 1000
gamma = 1.0
print_every = 1


class continuous_mountain_car(game):
    def __init__(self):
        self.gamma = 1.0
        self.number_of_sampled_trajectories = 0 # total number of sampled trajectories

        self.reset()

    def reset(self):
        global env
        # TODO: perform reset of policy networks
        torch.manual_seed(0)
        env.seed(0)
        env = gym.make('MountainCarContinuous-v0')
        self.snapshot_policy = GaussianPolicy() # policy "snapshot" network used by some algorithms
        self.policy = GaussianPolicy() # policy network parameters
        self.sample_policy = GaussianPolicy() # sample policy used during evaluation

    def sample(self, max_t = 1000, eval = 0):
        """
        sample a trajectory
        {state, action, log_prob, reward}
        snaphsot = True iff we sample from snapshot policy
        snapshot = False iff we sample from current policy
        max_t - maximum length of the trajectory
        """

        # If in evaluation mode, random sample
        if eval:
            policy = self.sample_policy
        else:
            policy = self.policy
        states = []
        actions = []
        saved_log_probs = []
        rewards = []
        state = env.reset()
        # Collect trajectory
        for t in range(max_t):
            states.append(state)
            action, log_prob = policy.act(state)
            actions.append(action)
            saved_log_probs.append(log_prob)
            next_state, reward, done, _ = env.step(action)

            # if next_state[0] - state[0] > 0 and action > 0:
            #     reward = 1
            # if next_state[0] - state[0] < 0 and action < 0:
            #     reward = 1
            state = next_state.copy()

            rewards.append(reward) # or after break? reward of terminal state?
            if done:
                # print(f"Done {t}", rewards, actions)

                break
        trajectory = {'states': states, 'actions': actions,
                        'probs': saved_log_probs, 'rewards': rewards}

        if self.number_of_sampled_trajectories % print_every == 0:
            print(sum(rewards))
        self.number_of_sampled_trajectories += 1
        return trajectory

    def evaluate(self): # performs the evaluation of the current policy NN
        # def evaluate(self, number_of_runs = 30):
        number_of_sampled_trajectories = self.number_of_sampled_trajectories
        results = self.sample(200,eval=1)['rewards']
        # results = [np.sum(self.sample(200, eval = 1)['rewards']) for i in range(number_of_runs)]
        self.number_of_sampled_trajectories = number_of_sampled_trajectories

        # TODO:
        # it should return 3 values:
        # 1) self.number_of_sampled_trajectories
        # 2) mean performance
        # 3) confidence interval
        return np.sum(self.sample(200,eval=1)['rewards'])
        # return (self.number_of_sampled_trajectories,np.mean(results),np.std(results))

    def generate_data(self, estimator, number_of_sampled_trajectories = 10000, number_of_runs = 30, root_path="./"):
        """
        generate a file of 3d tuples: (number of sample trajectories, mean reward, CI)
        until it reaches the specified number of trajectories ("number_of_sampled_trajectories")
        """
        # trajectories = []
        # mean_reward = []
        # CI_reward = []
        results = []
        for _ in range(number_of_runs):
            self.reset()
            estimator_instance = estimator(self)
            evaluations = []
            while True:
                estimator_instance.step(self) # performs one step of update for the selected estimator
                                       # this can be one or more episodes
                # after policy NN updates, we need to evaluate this updated policy using self.evaluate()
                evaluations.append((self.number_of_sampled_trajectories,self.evaluate()))
                # TODO: store the returned values: trajectories, mean_reward, CI_reward in some file
                if self.number_of_sampled_trajectories > number_of_sampled_trajectories:
                    # print("finish",`${}`)
                    print(f'finish run {_+1} of {number_of_runs}')
                    self.number_of_sampled_trajectories = 0
                    results.append(evaluations)
                    break
        # print(np.array(results).shape)
        # store a numpy binary
        name = 'data-v2--'+type(self).__name__+'_'+type(estimator_instance).__name__+'.npy'
        file_path = os.path.join(root_path, name)
        # store a numpy binary
        np.save(file_path,np.array(results))