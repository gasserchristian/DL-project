import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn
from torch import optim
from optimizers import SVRG_optim, SVRG_Snapshot, SARAH_optim, SARAH_Snapshot
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class policy_network():
    def __init__(self, env):
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))

    def predict_actions(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


def discount_rewards(rewards_list, gamma=0.99):
    r = np.array([gamma ** i * rewards_list[i]
                  for i in range(len(rewards_list))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    # return r - np.mean(r)
    return r


def reinforce(environment, policy_estimator, num_episodes=2000,
              batch_size=10, gamma=0.99):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizer
    optimizer = optim.SGD(policy_estimator.network.parameters(),
                          lr=0.01)

    action_space = np.arange(environment.action_space.n)
    ep = 0
    while ep < num_episodes:
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        while done == False:
            # Get actions and convert to numpy array
            action_probs = policy_estimator.predict_actions(
                s_0).detach().numpy()
            action = np.random.choice(action_space,
                                      p=action_probs)
            s_1, r, done, _ = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # If done, batch data
            if done:
                batch_rewards.extend(discount_rewards(
                    rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(
                        batch_rewards)
                    # Actions are used as indices, must be
                    # LongTensor
                    action_tensor = torch.LongTensor(
                        batch_actions)

                    # Calculate loss
                    logprob = torch.log(
                        policy_estimator.predict_actions(state_tensor))
                    selected_logprobs = reward_tensor * \
                                        torch.gather(logprob, 1,
                                                     torch.unsqueeze(action_tensor, 1)).squeeze()
                    loss = -selected_logprobs.mean()

                    # Calculate gradients for the main policy estimator
                    loss.backward()
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                print("Episode : %d, Average return of last 100: %.2f" % (ep + 1, avg_rewards), end='\r')
                ep += 1
    return total_rewards


def gpomdp(environment, policy_estimator, num_episodes=2000,
           batch_size=10, gamma=0.99):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizer
    optimizer = optim.SGD(policy_estimator.network.parameters(),
                          lr=0.01)

    action_space = np.arange(environment.action_space.n)
    ep = 0
    while ep < num_episodes:
        s_0 = environment.reset()
        states = []
        rewards = []
        actions = []
        avg_rewards = []
        done = False
        while done == False:
            # Get actions and convert to numpy array
            action_probs = policy_estimator.predict_actions(
                s_0).detach().numpy()

            # sample an action according to the correct probabilities
            action = np.random.choice(action_space,
                                      p=action_probs)

            # taking a step
            s_1, r, done, _ = environment.step(action)

            # appending states, rewards, actions involved in the episode
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # calculating the state-dependent average rewards (baseline)
            avg_rewards.append(np.expand_dims(np.mean(rewards, axis=0), axis=0))

            # If done, batch data
            if done:

                # log the batch (discounted) rewards - baseline
                batch_rewards.extend(np.reshape(discount_rewards(
                    rewards, gamma), (-1,)) - np.reshape(np.asarray(avg_rewards), (-1,)))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()

                    # convert the batch states, rewards, actions to torch tensors
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(
                        batch_rewards)
                    action_tensor = torch.LongTensor(
                        batch_actions)

                    # Calculate loss
                    logprob = torch.log(
                        policy_estimator.predict_actions(state_tensor))
                    selected_logprobs = reward_tensor * \
                                        torch.gather(logprob, 1,
                                                     torch.unsqueeze(action_tensor, 1)).squeeze()
                    loss = -selected_logprobs.mean()

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                print("Episode : %d, Average return of last 100: %.2f" % (ep + 1, avg_rewards), end='\r')
                ep += 1

    return total_rewards


def SVRG(environment, environment_snapshot, policy_estimator, inner_policy, num_episodes=2000, num_inner_episodes=400,
         batch_size=10, gamma=0.99):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizerS
    optimizer = SVRG_optim(policy_estimator.network.parameters(),
                           lr=0.001)
    optimizer_snap = SVRG_Snapshot(inner_policy.network.parameters())

    action_space = np.arange(environment.action_space.n)

    # num of episodes
    ep = 0

    while ep < num_episodes:
        inner_it = 0
        total_rewards_snap = []
        batch_rewards_snap = []
        batch_actions_snap = []
        batch_states_snap = []
        batch_counter_snap = 1

        optimizer_snap.zero_grad()  # zero out the gradients accumulated by optimizer_snap in the beginning of each outer loop

        # inner loop to calculate mean gradient
        while inner_it < num_inner_episodes:
            s_0_snap = environment_snapshot.reset()
            states_snap = []
            rewards_snap = []
            actions_snap = []
            avg_rewards_snap = []
            done_snap = False
            while done_snap == False:
                # Get actions and convert to numpy array
                action_probs_snap = inner_policy.predict_actions(
                    s_0_snap).detach().numpy()

                # sample an action according to the correct probabilities
                action_snap = np.random.choice(action_space,
                                               p=action_probs_snap)

                # taking a step
                s_1_snap, r_snap, done_snap, _ = environment_snapshot.step(action_snap)

                # appending states, rewards, actions involved in the episode
                states_snap.append(s_0_snap)
                rewards_snap.append(r_snap)
                actions_snap.append(action_snap)
                s_0_snap = s_1_snap

                # calculating the state-dependent average rewards (baseline)
                avg_rewards_snap.append(np.expand_dims(np.mean(rewards_snap, axis=0), axis=0))

                # If done, batch data
                if done_snap:

                    # log the batch (discounted) rewards - baseline
                    batch_rewards_snap.extend(np.reshape(discount_rewards(
                        rewards_snap, gamma), (-1,)) - np.reshape(np.asarray(avg_rewards_snap), (-1,)))
                    batch_states_snap.extend(states_snap)
                    batch_actions_snap.extend(actions_snap)
                    batch_counter_snap += 1
                    total_rewards_snap.append(sum(rewards_snap))

                    # If batch is complete, update target network
                    if batch_counter_snap == batch_size:
                        # convert the batch states, rewards, actions to torch tensors
                        state_tensor_snap = torch.FloatTensor(batch_states_snap)
                        reward_tensor_snap = torch.FloatTensor(
                            batch_rewards_snap)
                        action_tensor_snap = torch.LongTensor(
                            batch_actions_snap)

                        # Calculate loss
                        logprob_snap = torch.log(
                            inner_policy.predict_actions(state_tensor_snap))
                        selected_logprobs_snap = reward_tensor_snap * \
                                                 torch.gather(logprob_snap, 1,
                                                              torch.unsqueeze(action_tensor_snap, 1)).squeeze()
                        loss_snap = -selected_logprobs_snap.mean()

                        # Calculate gradients
                        loss_snap.backward()

                        batch_rewards_snap = []
                        batch_actions_snap = []
                        batch_states_snap = []
                        batch_counter_snap = 1
                    inner_it += 1

        # pass the current parameters of snap_optimizer to main SVRG optimizer
        u = optimizer_snap.get_param_groups()
        optimizer.set_u(u)

        s_0 = environment.reset()
        states = []
        rewards = []
        actions = []
        avg_rewards = []
        done = False
        while done == False:
            # Get actions and convert to numpy array
            action_probs = policy_estimator.predict_actions(
                s_0).detach().numpy()

            # sample an action according to the correct probabilities
            action = np.random.choice(action_space,
                                      p=action_probs)

            # taking a step
            s_1, r, done, _ = environment.step(action)

            # appending states, rewards, actions involved in the episode
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # calculating the state-dependent average rewards (baseline)
            avg_rewards.append(np.expand_dims(np.mean(rewards, axis=0), axis=0))

            # If done, batch data
            if done:
                # log the batch (discounted) rewards - baseline
                batch_rewards.extend(np.reshape(discount_rewards(
                    rewards, gamma), (-1,)) - np.reshape(np.asarray(avg_rewards), (-1,)))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()

                    # convert the batch states, rewards, actions to torch tensors
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(
                        batch_rewards)
                    action_tensor = torch.LongTensor(
                        batch_actions)

                    # Calculate loss
                    logprob = torch.log(
                        policy_estimator.predict_actions(state_tensor))



                    selected_logprobs = reward_tensor * \
                                        torch.gather(logprob, 1,
                                                     torch.unsqueeze(action_tensor, 1)).squeeze()
                    loss = -selected_logprobs.mean()

                    # Calculate gradients
                    loss.backward()

                    # see model_snap outputs and backpropagate the gradients
                    logprob_1 = torch.log(
                        inner_policy.predict_actions(state_tensor))
                    selected_logprobs_1 = reward_tensor * \
                                          torch.gather(logprob_1, 1,
                                                       torch.unsqueeze(action_tensor, 1)).squeeze()
                    loss_1 = -selected_logprobs_1.mean()
                    optimizer_snap.zero_grad()
                    loss_1.backward()

                    # Apply gradients
                    optimizer.step(optimizer_snap.get_param_groups())
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                print("Episode : %d, Average return of last 100: %.2f" % (ep + 1, avg_rewards), end='\r')
                ep += 1

        # update the snapshot params
        optimizer_snap.set_param_groups(optimizer.get_param_groups())

    return total_rewards


def SVRG_1(environment, environment_snapshot, policy_estimator, inner_policy, batch_size = 100, num_epochs = 50, mini_batch_size = 10, epoch_size = 5, num_episodes=2000, num_inner_episodes=400, gamma=0.99):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1



    # Define optimizerS
    optimizer = SVRG_optim(policy_estimator.network.parameters(),
                           lr=0.001)
    optimizer_snap = SVRG_Snapshot(inner_policy.network.parameters())

    action_space = np.arange(environment.action_space.n)


    # epoch

    for _ in np.arange(0,num_epochs):
        # inner loop to calculate mean gradient
        inner_it = 0
        while inner_it < batch_size:
            s_0_snap = environment_snapshot.reset()
            states_snap = []
            rewards_snap = []
            actions_snap = []
            avg_rewards_snap = []
            done_snap = False
            while done_snap == False:
                # Get actions and convert to numpy array
                action_probs_snap = inner_policy.predict_actions(
                    s_0_snap).detach().numpy()

                # sample an action according to the correct probabilities
                action_snap = np.random.choice(action_space,
                                               p=action_probs_snap)

                # taking a step
                s_1_snap, r_snap, done_snap, _ = environment_snapshot.step(action_snap)

                # appending states, rewards, actions involved in the episode
                states_snap.append(s_0_snap)
                rewards_snap.append(r_snap)
                actions_snap.append(action_snap)
                s_0_snap = s_1_snap

                # calculating the state-dependent average rewards (baseline)
                avg_rewards_snap.append(np.expand_dims(np.mean(rewards_snap, axis=0), axis=0))

                # If done, batch data
                if done_snap:

                    # log the batch (discounted) rewards - baseline
                    batch_rewards_snap.extend(np.reshape(discount_rewards(
                        rewards_snap, gamma), (-1,)) - np.reshape(np.asarray(avg_rewards_snap), (-1,)))
                    batch_states_snap.extend(states_snap)
                    batch_actions_snap.extend(actions_snap)
                    batch_counter_snap += 1
                    total_rewards_snap.append(sum(rewards_snap))

                    # If batch is complete, update target network
                    if batch_counter_snap == batch_size:
                        # convert the batch states, rewards, actions to torch tensors
                        state_tensor_snap = torch.FloatTensor(batch_states_snap)
                        reward_tensor_snap = torch.FloatTensor(
                            batch_rewards_snap)
                        action_tensor_snap = torch.LongTensor(
                            batch_actions_snap)

                        # Calculate loss
                        logprob_snap = torch.log(
                            inner_policy.predict_actions(state_tensor_snap))
                        selected_logprobs_snap = reward_tensor_snap * \
                                                 torch.gather(logprob_snap, 1,
                                                              torch.unsqueeze(action_tensor_snap, 1)).squeeze()
                        loss_snap = -selected_logprobs_snap.mean()

                        # Calculate gradients
                        loss_snap.backward()

                        batch_rewards_snap = []
                        batch_actions_snap = []
                        batch_states_snap = []
                        batch_counter_snap = 1
                    inner_it += 1
        # pass the current parameters of snap_optimizer to main SVRG optimizer
        u = optimizer_snap.get_param_groups()
        optimizer.set_u(u)

        # num of episodes
        ep = 0

        while ep < epoch_size:
            total_rewards_snap = []
            batch_rewards_snap = []
            batch_actions_snap = []
            batch_states_snap = []
            batch_counter_snap = 1

            optimizer_snap.zero_grad()  # zero out the gradients accumulated by optimizer_snap in the beginning of each outer loop

            s_0 = environment.reset()
            states = []
            rewards = []
            actions = []
            avg_rewards = []
            done = False
            while done == False:
                # Get actions and convert to numpy array
                action_probs = policy_estimator.predict_actions(
                    s_0).detach().numpy()

                # sample an action according to the correct probabilities
                action = np.random.choice(action_space,
                                          p=action_probs)

                # taking a step
                s_1, r, done, _ = environment.step(action)

                # appending states, rewards, actions involved in the episode
                states.append(s_0)
                rewards.append(r)
                actions.append(action)
                s_0 = s_1

                # calculating the state-dependent average rewards (baseline)
                avg_rewards.append(np.expand_dims(np.mean(rewards, axis=0), axis=0))

                # If done, batch data
                if done:
                    # log the batch (discounted) rewards - baseline
                    batch_rewards.extend(np.reshape(discount_rewards(
                        rewards, gamma), (-1,)) - np.reshape(np.asarray(avg_rewards), (-1,)))
                    batch_states.extend(states)
                    batch_actions.extend(actions)
                    batch_counter += 1
                    total_rewards.append(sum(rewards))

                    # If batch is complete, update network
                    if batch_counter == mini_batch_size:
                        optimizer.zero_grad()

                        # convert the batch states, rewards, actions to torch tensors
                        state_tensor = torch.FloatTensor(batch_states)
                        reward_tensor = torch.FloatTensor(
                            batch_rewards)
                        action_tensor = torch.LongTensor(
                            batch_actions)

                        # Calculate loss
                        logprob = torch.log(
                            policy_estimator.predict_actions(state_tensor))



                        selected_logprobs = reward_tensor * \
                                            torch.gather(logprob, 1,
                                                         torch.unsqueeze(action_tensor, 1)).squeeze()
                        loss = -selected_logprobs.mean()

                        # Calculate gradients
                        loss.backward()

                        # see model_snap outputs and backpropagate the gradients
                        optimizer_snap.zero_grad()
                        logprob_1 = torch.log(
                            inner_policy.predict_actions(state_tensor))
                        selected_logprobs_1 = reward_tensor * \
                                              torch.gather(logprob_1, 1,
                                                           torch.unsqueeze(action_tensor, 1)).squeeze()
                        loss_1 = -selected_logprobs_1.mean()
                        loss_1.backward()

                        # Apply gradients
                        optimizer.step(optimizer_snap.get_param_groups())
                        batch_rewards = []
                        batch_actions = []
                        batch_states = []
                        batch_counter = 1

                    avg_rewards = np.mean(total_rewards[-100:])
                    # Print running average
                    print("Episode : %d, Average return of last 100: %.2f" % (ep + 1, avg_rewards), end='\r')
                    ep += 1

            # update the snapshot params
            optimizer_snap.set_param_groups(optimizer.get_param_groups())

    return total_rewards

def SARAH(environment, policy_estimator, inner_policy, num_episodes=2000, num_inner_episodes=400,
         batch_size=10, gamma=0.99):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizers
    optimizer = SARAH_optim(policy_estimator.network.parameters(),
                           lr=0.001)
    optimizer_snap = SARAH_Snapshot(inner_policy.network.parameters())
    action_space = np.arange(environment.action_space.n)

    # num of episodes
    ep = 0

    while ep < num_episodes:
        inner_it = 0
        total_rewards_snap = []
        batch_rewards_snap = []
        batch_actions_snap = []
        batch_states_snap = []
        batch_counter_snap = 1

        optimizer_snap.zero_grad()  # zero out the gradients accumulated by optimizer_snap in the beginning of each outer loop

        # inner loop to calculate mean gradient
        while inner_it < num_inner_episodes:
            s_0_snap = environment.reset()
            states_snap = []
            rewards_snap = []
            actions_snap = []
            avg_rewards_snap = []
            done_snap = False
            while done_snap == False:
                # Get actions and convert to numpy array
                action_probs_snap = inner_policy.predict_actions(
                    s_0_snap).detach().numpy()

                # sample an action according to the correct probabilities
                action_snap = np.random.choice(action_space,
                                               p=action_probs_snap)

                # taking a step
                s_1_snap, r_snap, done_snap, _ = environment.step(action_snap)

                # appending states, rewards, actions involved in the episode
                states_snap.append(s_0_snap)
                rewards_snap.append(r_snap)
                actions_snap.append(action_snap)
                s_0_snap = s_1_snap

                # calculating the state-dependent average rewards (baseline)
                avg_rewards_snap.append(np.expand_dims(np.mean(rewards_snap, axis=0), axis=0))

                # If done, batch data
                if done_snap:

                    # log the batch (discounted) rewards - baseline
                    batch_rewards_snap.extend(np.reshape(discount_rewards(
                        rewards_snap, gamma), (-1,)) - np.reshape(np.asarray(avg_rewards_snap), (-1,)))
                    batch_states_snap.extend(states_snap)
                    batch_actions_snap.extend(actions_snap)
                    batch_counter_snap += 1
                    total_rewards_snap.append(sum(rewards_snap))

                    # If batch is complete, update target network
                    if batch_counter_snap == batch_size:
                        # convert the batch states, rewards, actions to torch tensors
                        state_tensor_snap = torch.FloatTensor(batch_states_snap)
                        reward_tensor_snap = torch.FloatTensor(
                            batch_rewards_snap)
                        action_tensor_snap = torch.LongTensor(
                            batch_actions_snap)

                        # Calculate loss
                        logprob_snap = torch.log(
                            inner_policy.predict_actions(state_tensor_snap))
                        selected_logprobs_snap = reward_tensor_snap * \
                                                 torch.gather(logprob_snap, 1,
                                                              torch.unsqueeze(action_tensor_snap, 1)).squeeze()
                        loss_snap = -selected_logprobs_snap.mean()

                        # Calculate gradients
                        loss_snap.backward()

                        batch_rewards_snap = []
                        batch_actions_snap = []
                        batch_states_snap = []
                        batch_counter_snap = 1
                    inner_it += 1

        # pass the current parameters of snap_optimizer to main SARAH optimizer
        v = optimizer_snap.get_param_groups()
        #
        optimizer.prev_param_groups = optimizer_snap.get_param_groups()
        optimizer.set_v(v)

        # initialize_previous parameter

        s_0 = environment.reset()
        states = []
        rewards = []
        actions = []
        avg_rewards = []
        done = False
        while done == False:
            # Get actions and convert to numpy array
            action_probs = policy_estimator.predict_actions(
                s_0).detach().numpy()

            # sample an action according to the correct probabilities
            action = np.random.choice(action_space,
                                      p=action_probs)

            # taking a step
            s_1, r, done, _ = environment.step(action)

            # appending states, rewards, actions involved in the episode
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # calculating the state-dependent average rewards (baseline)
            avg_rewards.append(np.expand_dims(np.mean(rewards, axis=0), axis=0))

            # If done, batch data
            if done:
                # log the batch (discounted) rewards - baseline
                batch_rewards.extend(np.reshape(discount_rewards(
                    rewards, gamma), (-1,)) - np.reshape(np.asarray(avg_rewards), (-1,)))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()

                    # convert the batch states, rewards, actions to torch tensors
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(
                        batch_rewards)
                    action_tensor = torch.LongTensor(
                        batch_actions)

                    # Calculate loss
                    logprob = torch.log(
                        policy_estimator.predict_actions(state_tensor))



                    selected_logprobs = reward_tensor * \
                                        torch.gather(logprob, 1,
                                                     torch.unsqueeze(action_tensor, 1)).squeeze()
                    loss = -selected_logprobs.mean()

                    # Calculate gradients
                    loss.backward()



                    # see model_snap outputs and backpropagate the gradients
                    logprob_1 = torch.log(
                        inner_policy.predict_actions(state_tensor))
                    selected_logprobs_1 = reward_tensor * \
                                          torch.gather(logprob_1, 1,
                                                       torch.unsqueeze(action_tensor, 1)).squeeze()
                    loss_1 = -selected_logprobs_1.mean()
                    optimizer_snap.zero_grad()
                    loss_1.backward()


                    # Apply gradients
                    optimizer.step(optimizer_snap.get_param_groups())

                    # update the prev params to the current_ones
                    optimizer_snap.set_param_groups(optimizer.prev_param_groups)

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1


                avg_rewards = np.mean(total_rewards[-100:])
                # optimizer_snap.set_param_groups(optimizer.param_groups)
                # Print running average
                print("Episode : %d, Average return of last 100: %.2f" % (ep + 1, avg_rewards), end='\r')
                ep += 1


    return total_rewards


env = gym.make('CartPole-v0')
env_snap = gym.make('CartPole-v0')
policy_est = policy_network(env)
inn_policy = policy_network(env)
# rewards = gpomdp(env, policy_est)
# rewards = reinforce(env, policy_est)
# rewards_result = SVRG(env, env_snap, policy_est, inn_policy)
# rewards_result = SARAH(env, policy_est)

rewards_result = SARAH(env, policy_est, inn_policy)

plt.plot(rewards_result)
# plt.savefig('SVRG.png')
plt.show()
