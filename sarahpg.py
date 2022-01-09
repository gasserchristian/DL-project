from operator import add

from cart_pole import Policy
from estimator import VrEstimator


class SarahPg(VrEstimator):

    # define here snapshot and current NNs
    def __init__(self, S=1000, m=10, lr=0.01, N=200, B=100):
        self.S = S  # number of epochs
        self.m = m  # epoch size
        self.N = N  # batch size
        self.B = B  # mini-batch size

        self.s = 0  # counter of epochs
        self.t = self.m  # counter within epoch

        self.mu = None  # return of outer loop

        self.current_policy = Policy()  # policy network
        self.snapshot_policy = Policy()  # snapshot neural network

        self.lr = lr  # learning rate

    def step(self, game):  # one step of update
        if self.t == self.m:
            self.outer_loop_update(game)  # outer loop of SARAH-PG algorithm
            self.t = 0  # reset counter within epoch

        self.inner_loop_update(game)  # inner loop of SARAH-PG algorithm
        self.t += 1

    def outer_loop_update(self, game):

        self.snapshot_update(game)  # update snapshot with weights of current NN

        gradient_estimators = self.gpomd_estimators(self.N,
                                                    game)  # then we sample a batch of trajectories and compute GPOMDP gradient estimators

        self.mu = [x / self.N for x in gradient_estimators]  # and average them out

    def inner_loop_update(self, game):
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

        ####################

        self.network_update(gradient_estimators, game)  # then we update current policy network
