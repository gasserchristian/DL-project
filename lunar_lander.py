"""
The implementation of the lunar_lander
"""

from game import game

import numpy as np
import scipy.signal
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.distributions.categorical import Categorical
import random
import numpy as np
import os

from LunarLander import LunarLander # we use customized version of LunarLander game  
from policies import Lunar_Policy # import policy network 


def discount_cumsum(x, discount):
    """
    Compute  cumulative sums of vectors.

    Input: [x0, x1, ..., xn]
    Output: [x0 + discount * x1 + discount^2 * x2 + ... , x1 + discount * x2 + ... , ... , xn]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def combined_shape(length, shape=None):
    """Helper function that combines two array shapes."""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class VPGBuffer:
    """
    Temporary buffer which we use to store trajectory
    """
    def __init__(self, obs_dim, act_dim, size, gamma):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        # advantage estimates
        self.phi_buf = np.zeros(size, dtype=np.float32)
        # rewards
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # trajectory's remaining return
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # log probabilities of chosen actions under behavior policy
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, logp):
        """
        Append a single timestep to the buffer. This is called at each environment
        update to store the outcome observed.
        """
        # buffer has to have room so you can store
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def end_traj(self, last_val=0):
        """
        Call after a trajectory ends. Last value is value(state) if cut-off at a
        certain state, or 0 if trajectory ended uninterrupted
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        self.ret_buf[path_slice] = discount_cumsum(rews[:-1], self.gamma) # reward-to-go 
        self.path_start_idx = self.ptr


    def get(self):
        """
        Call after an epoch ends. Resets pointers and returns the buffer contents.
        """
        # Buffer has to be full before you can get something from it.
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    phi=self.phi_buf, logp=self.logp_buf)

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

########################################################################################################

print_every = 2
store_every = 10
class lunar_lander(game):
    def __init__(self):
        super(lunar_lander, self).__init__()
        self.gamma = 0.99

        # Learning rates for policy 
        pi_lr = 3e-3
        
        self.env = LunarLander()
        self.reset()


    def reset(self, seed=42):
        self.reset_seeds(seed)
        self.snapshot_policy = Lunar_Policy()
        self.policy = Lunar_Policy()
        self.sample_policy = Lunar_Policy()
  

    def sample(self):
        obs_dim = [8]
        act_dim = []
        steps_per_epoch = 3000

        # The longest an episode can go on before cutting it off
        max_ep_len = 300

        # Discount factor for weighting future rewards
        gamma = self.gamma

        # Set up buffer
        buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma)

        # Initialize the environment
        state, ep_ret, ep_len = self.env.reset(), 0, 0

        ep_returns = []
        for t in range(steps_per_epoch):
            a, logp = self.policy.step(torch.as_tensor(state, dtype=torch.float32)) # compute action and its log-probability 
            next_state, r, terminal = self.env.transition(a) # next state, reward, whether next state is terminal or not
            
            ep_ret += r # return of the episode 
            ep_len += 1

            # Log transition in buffer
            buf.store(state, a, r, logp)

            # Update state (critical!)
            state = next_state

            timeout = ep_len == max_ep_len
            epoch_ended = (t == steps_per_epoch - 1)

            if terminal or timeout or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    _, _ = self.policy.step(torch.as_tensor(state, dtype=torch.float32))
                else:
                    v = 0
                if timeout or terminal:
                    ep_returns.append(ep_ret)  # only store return when episode ended
                buf.end_traj(0)
                state, ep_ret, ep_len = self.env.reset(), 0, 0

        mean_return = np.mean(ep_returns) if len(ep_returns) > 0 else np.nan
        
        # Extract data from buffer
        data = buf.get()
        ret = data['ret']
        obs = data['obs'] # observations
        act = data['act'] # actions 
        log_p = self.policy.forward(obs,act)[1] # logp 

        trajectory = {'states': obs, 'actions': act, 'probs': log_p, 'rewards': ret}

        if self.number_of_sampled_trajectories % print_every == 0:
            print(f"mean return {mean_return}", f"of {self.number_of_sampled_trajectories} trajectory")

        if self.number_of_sampled_trajectories % store_every == 0:
            self.rewards_buffer.append(mean_return)

        self.number_of_sampled_trajectories += 1
        return trajectory 




    