"""
The implementation of "Lunar Lander" game class
"""

import numpy as np
import scipy.signal
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.distributions.categorical import Categorical
import random
import os

from game import game
from LunarLander import LunarLander # we use customized version of LunarLander game  
from policies import Lunar_Policy # import policy network 
from PG_Buffer import PG_Buffer # import policy buffer 


print_every = 2 # print every second trajectory  
store_every = 1 # every trajectory is stored in the file 


class lunar_lander(game):
    def __init__(self):
        super(lunar_lander, self).__init__()
        self.gamma = 0.99 # discount factor for the game 
        self.env = LunarLander() # initialize game environment 
        self.reset() # perform reset of policy networks 


    def reset(self, seed=42):
        """ resets seeds and policies """
        self.reset_seeds(seed)
        self.snapshot_policy = Lunar_Policy()
        self.policy = Lunar_Policy()
  

    def sample(self):
        """ samples trajectory and return it """ 
        
        obs_dim = [8]
        act_dim = []
        steps_per_epoch = 3000

        # The longest an episode can go on before cutting it off
        max_ep_len = 300

        # Discount factor for weighting future rewards
        gamma = self.gamma

        # Set up buffer
        buf = PG_Buffer(obs_dim, act_dim, steps_per_epoch, gamma)

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
                if timeout or terminal:
                    ep_returns.append(ep_ret)  # only store return when episode ended
                buf.end_traj(0)
                state, ep_ret, ep_len = self.env.reset(), 0, 0

        mean_return = np.mean(ep_returns) if len(ep_returns) > 0 else np.nan
        
        # Extract data from buffer
        data = buf.get()
        ret = data['ret'] # rewards to go
        obs = data['obs'] # observations
        act = data['act'] # actions 
        log_p = self.policy.forward(obs,act)[1] # forward pass, get log probabilities  

        trajectory = {'states': obs, 'actions': act, 'probs': log_p, 'rewards': ret} # store trajectory as a dictionary  

        # if self.number_of_sampled_trajectories % print_every == 0:
        #     print(f"mean return {mean_return}", f"of {self.number_of_sampled_trajectories} trajectory")

        if self.number_of_sampled_trajectories % store_every == 0:
            self.rewards_buffer.append(mean_return) # this buffer is used for writing files after training 

        self.number_of_sampled_trajectories += 1
        return trajectory 




    