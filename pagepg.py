from operator import add

import numpy as np

from cart_pole import Policy
from estimator import VrEstimator


class PagePg(VrEstimator):

    # define here snapshot and current NNs
    def __init__(self, S=1000, m=10, lr=0.01, prob=None, N=200, B=100):
        self.S = S  # number of epochs
        self.m = m  # epoch size
        self.N = N  # batch size
        self.B = B  # mini-batch size
        if prob is None:
            self.prob = self.B / (self.N + self.B)  # switching probability

        self.s = 0  # counter of epochs
        self.t = self.m  # counter within epochs
        self.p = 1  # if 1, compute full gradient calculation; if 0, do SARAH
        self.mu = None  # return of outer loop

        self.current_policy = Policy()  # policy network
        self.snapshot_policy = Policy()  # snapshot neural network

        self.lr = lr  # learning rate

    def step(self, game):  # one step of update
        if self.p:
            self.full_grad_update(game)  # full grad calculation of PAGE-PG algorithm
        else:
            self.sarah_inner_update(game)  # inner loop of PAGE-PG algorithm
        self.t += 1  # update counter for step updates
        self.p = np.random.choice(2, p=[1 - self.prob, self.prob])

    def full_grad_update(self, game):

        self.snapshot_policy

        """
		then we sample a batch of trajectories and compute GPOMDP gradient estimation
		"""

        for i in range(self.N):  # sample a batch of trajectories
            trajectory = game.sample()  # trajectory is produced by current policy NN
            gradient_estimator = self.gradient_estimate(trajectory, game)  # compute gradient estimate
            if i == 0:
                gradient_estimators = gradient_estimator
                continue
            gradient_estimators = list(map(add, gradient_estimators, gradient_estimator))  # and then we sum them up

        self.mu = [x / self.N for x in gradient_estimators]  # and average them out
        self.network_update(self.mu, game)  # then we update current policy network

    def sarah_inner_update(self, game):
        gradient_estimators = []

        for i in range(self.B):
            trajectory = game.sample()  # trajectory produced by current policy network

            gradient_estimator = self.gradient_estimate(trajectory, game, snapshot=False)
            snapshot_estimator = self.gradient_estimate(trajectory, game, snapshot=True)

            weight = self.importance_weight(trajectory, game)  # TODO
            gradient_estimator = [x / self.B for x in gradient_estimator]
            to_add = [x / self.B * (-1) * weight + self.mu[i] for i, x in enumerate(snapshot_estimator)]

            if i == 0:
                gradient_estimators = list(map(add, gradient_estimator, to_add))
                continue
            gradient_estimators = list(map(add, gradient_estimators, to_add))

        # Update the stochastic step direction mu recursively
        self.mu = gradient_estimators

        self.snapshot_update(game)  # update snapshot parameters to current_policy_network

        self.network_update(gradient_estimators, game)  # then we update current policy network
