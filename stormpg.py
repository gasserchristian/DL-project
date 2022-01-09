from operator import add

from cart_pole import Policy
from estimator import VrEstimator


class StormPg(VrEstimator):

    # define here snapshot and current NNs
    def __init__(self, S=1000, m=10, lr=0.01, alpha=0.5, N=200, B=100):
        self.S = S  # number of epochs
        self.m = m  # epoch size
        self.N = N  # batch size
        self.B = B  # mini-batch size
        self.alpha = alpha  # weighing STORM hyperparameter in the range [0,1]

        self.s = 0  # counter of epochs
        self.t = self.m  # counter within epoch

        self.mu = None  # return of outer loop

        self.current_policy = Policy()  # policy network
        self.snapshot_policy = Policy()  # snapshot neural network

        self.lr = lr  # learning rate

    def importance_weight(self, trajectory, game):
        # TODO: compute importance weight for trajectory between
        # current and old policy network
        return 1

    def step(self, game):  # one step of update
        if self.t == self.m:
            self.outer_loop_update(game)  # outer loop of STORM-PG algorithm
            self.t = 0  # reset counter within epoch

        self.inner_loop_update(game)  # inner loop of STORM-PG algorithm
        self.t += 1

    def outer_loop_update(self, game):
        self.snapshot_update(game)
        gradient_estimators = self.gpomd_estimators(self.N,
                                                    game)  # then we sample a batch of trajectories and compute GPOMDP gradient estimation

        self.mu = [x / self.N for x in gradient_estimators]  # and average them out

    def inner_loop_update(self, game):
        gradient_estimators = []

        for i in range(self.B):
            trajectory = game.sample()  # trajectory produced by current policy network

            gradient_estimator = self.gradient_estimate(trajectory, game, snapshot=False)
            snapshot_estimator = self.gradient_estimate(trajectory, game, snapshot=True)

            weight = self.importance_weight(trajectory, game)  # TODO
            gradient_estimator = [x / self.B for x in gradient_estimator]
            to_add = [(1 - self.alpha) * (x / self.B * (-1) * weight + self.mu[i]) for i, x in
                      enumerate(snapshot_estimator)]

            if i == 0:
                gradient_estimators = list(map(add, gradient_estimator, to_add))
                continue
            gradient_estimators = list(map(add, gradient_estimators, to_add))

        # Update the stochastic step direction mu recursively
        self.mu = gradient_estimators

        self.snapshot_update(game)  # update snapshot parameters to current_policy_network

        self.network_update(gradient_estimators, game)  # then we update current policy network
