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


from garage.torch.distributions import TanhNormal
from garage.torch.modules.mlp_module import MLPModule
from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule


# from garage.torch.modules import GaussianMLPModule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
env = gym.make('MountainCarContinuous-v0')
env.seed(0)

max_t = 1000
gamma = 1.0
print_every = 1


class Policy(nn.Module):
    def __init__(self,
                 input_dim =2,
                 output_dim = 1,
                 hidden_nonlinearity=nn.ReLU()):
        super().__init__()

        self.model = nn.Sequential(*[
            nn.Linear(input_dim, 16),
            hidden_nonlinearity,
            nn.Linear(16, 8),
            hidden_nonlinearity,
            nn.Linear(8, output_dim)
        ]).to(device)

        self.variance = torch.eye(output_dim, device=device) * 1e-3
        
    def forward(self, x):
        
        return torch.distributions.multivariate_normal.MultivariateNormal(
            self.model(x), covariance_matrix=self.variance)

    def act(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # probs = self.forward(state).cpu()
        # model = Categorical(probs)
        # action = model.sample()
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        dist = self.forward(state)
        

        # print(action)
        sample = dist.sample()
        sample = torch.clip(sample, -1, 1)
        log_prob = dist.log_prob(sample)

        return sample.cpu(), log_prob.cpu()

    def log_prob(self, state, action): # probability of taking action "action" in state "state"
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        log_prob = probs.log_prob(action)
        return log_prob.cpu()


class GaussianMLPBaseModule(nn.Module):
    """Base of GaussianMLPModel.
    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_hidden_w_init (callable):  Initializer function for the weight
            of hidden layer (s).
        std_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s).
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation.
            - softplus: the std will be computed as log(1+exp(x)).
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=torch.tanh,
                 std_hidden_w_init=nn.init.xavier_uniform_,
                 std_hidden_b_init=nn.init.zeros_,
                 std_output_nonlinearity=None,
                 std_output_w_init=nn.init.xavier_uniform_,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal):
        super().__init__()

        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = output_dim
        self._learn_std = learn_std
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_hidden_w_init = std_hidden_w_init
        self._std_hidden_b_init = std_hidden_b_init
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_output_w_init = std_output_w_init
        self._std_parameterization = std_parameterization
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._norm_dist_class = normal_distribution_cls

        if self._std_parameterization not in ('exp', 'softplus'):
            raise NotImplementedError

        init_std_param = torch.Tensor([init_std]).log()
        if self._learn_std:
            self._init_std = torch.nn.Parameter(init_std_param)
        else:
            self._init_std = init_std_param
            self.register_buffer('init_std', self._init_std)

        self._min_std_param = self._max_std_param = None
        if min_std is not None:
            self._min_std_param = torch.Tensor([min_std]).log()
            self.register_buffer('min_std_param', self._min_std_param)
        if max_std is not None:
            self._max_std_param = torch.Tensor([max_std]).log()
            self.register_buffer('max_std_param', self._max_std_param)

    # def to(self, *args, **kwargs):
    #     """Move the module to the specified device.
    #     Args:
    #         *args: args to pytorch to function.
    #         **kwargs: keyword args to pytorch to function.
    #     """
    #     super().to(*args, **kwargs)
    #     buffers = dict(self.named_buffers())
    #     if not isinstance(self._init_std, torch.nn.Parameter):
    #         self._init_std = buffers['init_std']
    #     # self._min_std_param = buffers['min_std_param']
    #     # self._max_std_param = buffers['max_std_param']

    @abc.abstractmethod
    def _get_mean_and_log_std(self, *inputs):
        pass

    def forward(self, *inputs):
        """Forward method.
        Args:
            *inputs: Input to the module.
        Returns:
            torch.distributions.independent.Independent: Independent
                distribution.
        """
        mean, log_std_uncentered = self._get_mean_and_log_std(*inputs)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()
        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist


class GaussianMLPModule(GaussianMLPBaseModule):
    """GaussianMLPModule that mean and std share the same network.
    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal):
        super(GaussianMLPModule,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_sizes=hidden_sizes,
                             hidden_nonlinearity=hidden_nonlinearity,
                             hidden_w_init=hidden_w_init,
                             hidden_b_init=hidden_b_init,
                             output_nonlinearity=output_nonlinearity,
                             output_w_init=output_w_init,
                             output_b_init=output_b_init,
                             learn_std=learn_std,
                             init_std=init_std,
                             min_std=min_std,
                             max_std=max_std,
                             std_parameterization=std_parameterization,
                             layer_normalization=layer_normalization,
                             normal_distribution_cls=normal_distribution_cls)

        self._mean_module = MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization)

    def _get_mean_and_log_std(self, *inputs):
        """Get mean and std of Gaussian distribution given inputs.
        Args:
            *inputs: Input to the module.
        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.
        """
        assert len(inputs) == 1
        mean = self._mean_module(*inputs)

        broadcast_shape = list(inputs[0].shape[:-1]) + [self._action_dim]
        uncentered_log_std = torch.zeros(*broadcast_shape).to(device) + self._init_std

        return mean, uncentered_log_std


class PolicyOLD(nn.Module):
    # neural network for the policy
    # TODO: change NN architecture to the optimal for the cart-pole game
    def __init__(self, state_size=2, action_size=1, hidden_size=32):
        super(Policy, self).__init__()
        # self._module = GaussianMLPModule(input_dim=state_size, 
        #     output_dim=action_size, 
        #     hidden_sizes=(16,16),
        #     hidden_nonlinearity=torch.relu,
        #     learn_std=True,
        #     init_std=1.0,
        #     min_std=1e-6,
        #     max_std=None,
        #     std_parameterization='exp',
        #     layer_normalization=False)
        # self._module.to(device)
        self._module = GaussianMLP(input_dim=state_size,output_dim=action_size)


    def forward(self, state):
        
        x = self._module(state)
        # we just consider 1 dimensional probability of action
        return x

    def act(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # probs = self.forward(state).cpu()
        # model = Categorical(probs)
        # action = model.sample()
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action = self.forward(state)
        # print(action)
        sample = action.sample()
        sample = torch.clip(sample, -1, 1)
        log_prob = action.log_prob(sample)

        return sample.cpu(), log_prob.cpu()

    def log_prob(self, state, action): # probability of taking action "action" in state "state"
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        log_prob = probs.log_prob(action)
        return log_prob.cpu()


class continuous_mountain_car(game):
    def __init__(self):
        self.gamma = 1.0
        self.number_of_sampled_trajectories = 0 # total number of sampled trajectories

        self.snapshot_policy = Policy() # policy "snapshot" network used by some algorithms
        self.policy = Policy() # policy network parameters
        self.sample_policy = Policy()# sample policy used during evaluation

    def reset(self):
        global env
        # TODO: perform reset of policy networks
        torch.manual_seed(0)
        env.seed(0)
        env = gym.make('MountainCarContinuous-v0')
        self.snapshot_policy = Policy()
        self.policy = Policy()
        self.sample_policy = Policy()

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
        np.save('data-runs--'+type(self).__name__+'_'+type(estimator_instance).__name__+'.npy',np.array(results))
        # np.savetxt('data-runs--'+type(self).__name__+'_'+type(estimator_instance).__name__+'.txt',np.array(results))
        # np.savetxt('data--'+type(self).__name__+'_'+type(estimator_instance).__name__+'.txt',np.array(evaluations).transpose())
