import numpy as np
from torch import optim
from estimator import VrEstimator

"""
Parameters to sweep over:
- batch size (try 5, 10, 20, 50, 100) - 5
- probability (try 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) - 9
- learning_rate (try 0.001, 0.005, 0.01, 0.05, 0.1) - 3
- alphas (try 0.1, 0.5, 0.9) - 3

800 jobs 
"""



class PageStormPg_wo_evaluation(VrEstimator):

    # define here snapshot and current NNs
    def __init__(self, game, N=100, B=10, prob=None, alpha=0.5)
        
        self.N = N  # batch size
        self.B = floor(N^(1/2))  # mini-batch size

        self.mu = None  # g_t*(1-alpha) 

        self.optimizer_sub = optim.Adam(game.policy.parameters(), lr=0.01)

        self.first_iteration_lr = 1  # this is magnitude of update by self.optimizer_first
        self.main_lr = 1  # this is magnitude of update by self.optimizer_sub

        if prob is None:
            self.prob = self.B / (self.N + self.B)  # switching probability
        self.p = 1  # sampled probability value: if 1, do full gradient calculation; if 0, do SARAH (initialize to 1)
        self.mu = None  # return of outer loop
        self.alpha = alpha
        self.policy_parameter_list = []

    def step(self, game):  # one step of update
        if self.p:
            self.full_grad_update(game)  # full grad calculation of PAGE-PG algorithm
        else:
            self.storm_inner_update(game)  # inner loop of PAGE-PG algorithm
        self.t += 1  # update counter for step updates
        self.p = np.random.choice(2, p=[1 - self.prob, self.prob])

        # sample randomly from learned policies
        # self.sample_policy_update(game)

    def full_grad_update(self, game):

        self.snapshot_update(game)  # update snapshot with weights of current NN
        gradient_estimators = self.outer_loop_estimators(
            game)  # then we sample a batch of trajectories and compute GPOMDP gradient estimators

        # self.mu is the main result of this method
        self.mu = {k: v / self.N for k, v in gradient_estimators.items()}  # average
        self.network_update(self.mu, game, first_iteration=False)  # then we update current policy network

        # rescale self.mu with (1-alpha)
        self.mu = {k: (1 - self.alpha) * v / self.N for k, v in gradient_estimators.items()}  # average and scale

    # average and scale
    def storm_inner_update(self, game):
        gradient_estimators = self.inner_loop_estimators(game)
        c = {k: v / self.B for k, v in gradient_estimators.items()}
        v = self.sum_dictionaries(c, self.mu)

        # Update the stochastic step direction mu recursively and scale accordingly
        self.mu = {k: (1 - self.alpha) * item for k, item in v.items()}

        self.network_update(v, game, first_iteration=False)  # updates network in remaining iterations
