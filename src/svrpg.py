from torch import optim

from estimator import VrEstimator
from adam import adam # our own adam optimizer

"""
Check the following (need thorough evaluation to verify the benefit)
1) return from time to time random policy 
2) enable vs disable weights 
2) enable vs disable normalization of weights
3) different initializations of ADAM 
4) normalized GPOMDP estimates vs unnormalized
5) run svrpg_manual vs svrpg_automatic  
"""


class Svrpg(VrEstimator):

    # define here snapshot and current NNs
    def __init__(self, game, args):
        # alpha = 1e-2,

        self.m = args["subit"]  # max allowed number of subiterations

        self.N = args["batch_size"]  # batch size
        self.B = args["mini_batch_size"]  # mini-batch size

        self.t = self.m  # counter within epoch

        self.mu = None  # return of outer loop (mean gradient calculated using N samples)

        #self.optimizer_first = adam(alpha = 0.003) # adam for first subiteration update 
        #self.optimizer_sub = adam(alpha = 0.003) # adam for remaining subiteration updates 

        self.optimizer_first = optim.Adam(game.policy.parameters(), lr=args["flr"])
        self.optimizer_sub = optim.Adam(game.policy.parameters(), lr=args["lr"])

        self.first_iteration_lr = 1  # this is magnitude of update by self.optimizer_first
        self.main_lr = 1  # this is magnitude of update by self.optimizer_sub
        self.policy_parameter_list = []

    def step(self, game):
        """
		This method performs a single update of parameters 
		It is the main method of SVRPG class
		"""

        if self.t == self.m:  # whenever the subiteration counter achieves threshold value
            self.outer_loop_update(game)  # outer loop update of SVRPG algprithm
            self.t = 0  # and reset the counter to 0

        self.inner_loop_update(game)  # inner loop update of SVRPG algprithm
        self.t += 1

        
        if (self.first_iteration_lr / self.N) > (self.main_lr / self.B):
            
			#Whenever this is true, we finish subiterations and take snapshot 
			#We need to verify that this indeed helps 
			
            self.t = self.m
        


    def outer_loop_update(self, game):

        self.snapshot_update(game)  # update snapshot with weights of current NN
        """
		then we sample B trajectories and compute gradient estimation
		"""
        gradient_estimators = self.outer_loop_estimators(game)

        # self.mu is the main result of this method
        self.mu = {k: v / self.N for k, v in gradient_estimators.items()}  # average


    def inner_loop_update(self, game):
        gradient_estimators = self.inner_loop_estimators(game)
        c = {k: v / self.B for k, v in gradient_estimators.items()}
        v = self.sum_dictionaries(c, self.mu)

        if self.t == 0:  # we use different adams for first subiteration and for the remaining
            self.network_update(v, game, first_iteration=True)  # updates network in first subiteration
        else:
            self.network_update(v, game, first_iteration=False)  # updates network in remaining iterations
