from estimator import VrEstimator
from torch import optim


class StormPg(VrEstimator):

    # define here snapshot and current NNs
    def __init__(self, game, m=50, N=100, B=10, alpha=0.5):
        self.m = m  # max allowed number of subiterations

        self.N = N  # batch size
        self.B = B  # mini-batch size

        self.t = self.m  # counter within epoch

        self.mu = None  # return of outer loop (mean gradient calculated using N samples)

        self.optimizer_first = optim.Adam(game.policy.parameters(), lr=0.02)
        self.optimizer_sub = optim.Adam(game.policy.parameters(), lr=0.01)

        self.first_iteration_lr = 1  # this is magnitude of update by self.optimizer_first
        self.main_lr = 1  # this is magnitude of update by self.optimizer_sub
        self.alpha = alpha

    def step(self, game):  # one step of update
        if self.t == self.m:
            self.outer_loop_update(game)  # outer loop of SARAH-PG algorithm
            self.t = 0  # reset counter within epoch

        self.inner_loop_update(game)  # inner loop of SARAH-PG algorithm
        self.t += 1

    def outer_loop_update(self, game):

        self.snapshot_update(game)  # update snapshot with weights of current NN
        gradient_estimators = self.outer_loop_estimators(
            game)  # then we sample a batch of trajectories and compute GPOMDP gradient estimators

        # self.mu is the main result of this method
        self.mu = {k: (1 - self.alpha) * v / self.N for k, v in gradient_estimators.items()}  # average and scale

    def inner_loop_update(self, game):
        gradient_estimators = self.inner_loop_estimators(game)
        c = {k: v / self.B for k, v in gradient_estimators.items()}
        v = self.sum_dictionaries(c, self.mu)

        # Update the stochastic step direction mu recursively and scale accordingly
        self.mu = {k: (1-self.alpha) * item for k, item in v.items()}

        # Update snapshot
        self.snapshot_update(game)

        if self.t == 0:  # we use different adams for first subiteration and for the remaining
            self.network_update(v, game, first_iteration=True)  # updates network in first subiteration
        else:
            self.network_update(v, game, first_iteration=False)  # updates network in remaining iterations
