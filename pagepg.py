from operator import add
import numpy as np
from torch import optim
from estimator import VrEstimator


class PagePg(VrEstimator):

    # define here snapshot and current NNs
    def __init__(self, game, m=50, N=100, B=10, prob=None):
        self.m = m  # max allowed number of subiterations

        self.N = N  # batch size
        self.B = B  # mini-batch size

        self.t = self.m  # counter within epoch

        self.mu = None  # return of outer loop (mean gradient calculated using N samples)

        self.optimizer_first = optim.Adam(game.policy.parameters(), lr=0.02)
        self.optimizer_sub = optim.Adam(game.policy.parameters(), lr=0.01)

        self.first_iteration_lr = 1  # this is magnitude of update by self.optimizer_first
        self.main_lr = 1  # this is magnitude of update by self.optimizer_sub

        if prob is None:
            self.prob = self.B / (self.N + self.B)  # switching probability
        self.p = 1  # sampled probability value: if 1, do full gradient calculation; if 0, do SARAH (initialize to 1)
        self.mu = None  # return of outer loop

    def step(self, game):  # one step of update
        if self.p:
            self.full_grad_update(game)  # full grad calculation of PAGE-PG algorithm
        else:
            self.sarah_inner_update(game)  # inner loop of PAGE-PG algorithm
        self.t += 1  # update counter for step updates
        self.p = np.random.choice(2, p=[1 - self.prob, self.prob])

    def full_grad_update(self, game):

        self.snapshot_update(game)  # update snapshot with weights of current NN
        gradient_estimators = self.outer_loop_estimators(
            game)  # then we sample a batch of trajectories and compute GPOMDP gradient estimators

        # self.mu is the main result of this method
        self.mu = {k: v / self.N for k, v in gradient_estimators.items()}  # average
        self.network_update(self.mu, game, first_iteration=True)  # then we update current policy network

    def sarah_inner_update(self, game):
        gradient_estimators = self.inner_loop_estimators(game)
        c = {k: v / self.B for k, v in gradient_estimators.items()}
        v = self.sum_dictionaries(c, self.mu)

        # Update the stochastic step direction mu recursively
        self.mu = v

        # Update snapshot
        self.snapshot_update(game)

        self.network_update(gradient_estimators, game, first_iteration=False)  # then we update current policy network
