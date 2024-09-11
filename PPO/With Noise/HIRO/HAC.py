#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import Tensor
from torch.nn import functional
import numpy as np
from DDPG import DDPG_High
from DDPG import DDPG_Low
from DDPG import Actor_Low
from utils import ReplayBuffer_Low
from utils import ReplayBuffer_High
#from CSTR import norm_action
#import  copy
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")






class HAC:
    def __init__(self, k_level, policy_freq, tau, c, state_dim, action_dim, goal_dim, goal_index, goal, render, threshold,
                 action_offset, state_offset, action_bounds, state_bounds, max_goal, lr):

        # adding the lowest level
        self.HAC = [DDPG_Low(state_dim, action_dim, goal_dim, action_bounds, action_offset, policy_freq, tau, lr)]
        self.replay_buffer = [ReplayBuffer_Low()]

        # adding remaining levels
        for _ in range(k_level-1):
            self.HAC.append(DDPG_High(state_dim, goal_dim, goal_index, state_bounds, state_offset, policy_freq, tau, lr))
            self.replay_buffer.append(ReplayBuffer_High())
        
        # set some parameters
        self.goal_dim = goal_dim
        self.goal_index = goal_index
        self.max_goal = max_goal
        self.goal = goal
        self.k_level = k_level
        self.c = c
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.threshold = threshold
        self.render = render
        self.action_bounds = action_bounds
        self.action_offset = action_offset



        # logging parameters
        self.goals = [None]*self.k_level
        self.reward = 0
        self.lo = 0
        self.iae = 0
        self.timestep = 0
        self.propylene_glycol = []
        self.flowrate = []
        self.solved = False

    def set_parameters(self, lamda, gamma, n_iter, batch_size, action_clip_low, action_clip_high,
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise):
        
        self.lamda = lamda
        self.gamma = gamma
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.action_clip_low = action_clip_low
        self.action_clip_high = action_clip_high
        self.state_clip_low = state_clip_low
        self.state_clip_high = state_clip_high
        self.exploration_action_noise = exploration_action_noise
        self.exploration_state_noise = exploration_state_noise


    def off_policy_correction(self,actor, action_sequence, state_sequence, goal_dim, goal_index, goal, end_state, max_goal, device):
    # initialize
        action_sequence = torch.stack(action_sequence).to(device)
        state_sequence = torch.stack(state_sequence).to(device)
        max_goal = max_goal.cpu()
        # prepare candidates
        mean = (torch.from_numpy(end_state) - state_sequence[0])[goal_index].cpu().unsqueeze(0)
        std = 0.5 * max_goal
        candidates = [torch.min(torch.max(Tensor(np.random.normal(loc=mean, scale=std, size=goal_dim).astype(np.float32)), max_goal), -max_goal) for _ in range(3)]
        candidates.append(mean)
        candidates.append(torch.from_numpy(goal).cpu())
        # select maximal
        candidates = torch.stack(candidates).to(device)

        sequence = [state_sequence[0][goal_index] + candidate - state_sequence[:, goal_index] for candidate in candidates]

        sequence = torch.stack(sequence).float().t().unsqueeze(2)
        b = state_sequence



        surr_prob = [-functional.mse_loss(action_sequence, actor(state_sequence.to(torch.float32), sequence[:,candidate])) for candidate in range(len(candidates))]

        surr_prob = [t.detach().numpy() for t in surr_prob]

        index = int(np.argmax(surr_prob))

        updated = (index != 9)
        goal_hat = candidates[index].numpy()

        return goal_hat, updated

    

    def intrinsic_reward(self, state, goal, next_state):
        difference = goal + state[self.goal_index] - next_state[self.goal_index]
        #difference = goal  - next_state

        distance = ((difference ** 2).sum()) ** (1 / 2)
        reward = -(distance**2)

        return reward # one dimensional, difference will be the distance between states
        # return -torch.pow(sum(torch.pow(state + goal - next_state, 2)), 1 / 2)


    def dense_reward(self,state, goal):
        difference = abs(goal - state[self.goal_index])

        distance = ((difference ** 2).sum()) ** (1 / 2)
        reward = -(distance**2)
        #reward = -(difference.sum()) # one dimensional, difference will be the distance between states

        return reward


    def h_function(self, next_state, state, goal, goal_index):
        return state[goal_index] + goal - next_state[goal_index]


    def check_goal(self, state, goal, threshold):
        difference = abs(state[2] - goal)

        if difference > threshold:
            return False
        return True

    def norm_action(self,action):
        low = -1
        high = 1

        action = ((action - low) / (high - low))

        action = 78 + action

        return action
    
    
    def run_HAC(self, env, i_level, state, tot_time, test):

        time = 0.01
        dt = 0.05


        #show_goal_achieve = True
        final_goal = self.goal
        next_state = None
        max_goal = torch.from_numpy(np.array(self.max_goal))

        steps = 0 # number of steps taken by agent

        episode_reward_h = 0
        state_sequence, goal_sequence, action_sequence, intri_reward_sequence, reward_h_sequence = [], [], [], [], []
        goal = self.HAC[i_level].select_action_High(state)

       # goal = goal + np.random.normal(0, self.exploration_state_noise)

        goal = goal.clip(self.state_clip_low[self.goal_index] , self.state_clip_high[self.goal_index])

        #for t in range(step, max_timestep):

        while time < tot_time:
            steps = steps + 1
            action = self.HAC[i_level - 1].select_action_Low(state, goal)  # action taken by lower level policy
            #action = norm_action(action)

            action = action + np.random.normal(0, self.exploration_action_noise)
            action = action.clip(self.action_clip_low, self.action_clip_high)
            #action = np.array([0.000000000000000000001])

            # 2.2.2 interact environment
            #print("state", state)
            next_state, reward = env(action,time, state)

            self.lo += (np.abs(next_state[2] - final_goal) ** 2)
            self.iae += (np.abs(next_state[2] - final_goal))


            self.propylene_glycol.append(next_state[2])
            self.flowrate.append(action[0])

            # 2.2.3 compute step arguments
            #reward_h = self.dense_reward(state,final_goal)
            reward_h = reward

            intri_reward = self.intrinsic_reward(state, goal, next_state)
            self.reward += reward
            next_goal = self.h_function(next_state, state, goal, self.goal_index)

            #next_goal = next_goal.clip(self.state_clip_low[self.goal_index], self.state_clip_high[self.goal_index])

            #print("next goal shape",next_goal.shape)
            #print("goal", next_goal)
            # 2.2.4 collect low-level experience
            self.replay_buffer[i_level-1].add((state, action, goal, intri_reward, next_state, next_goal, self.gamma))

            state_sequence.append(torch.from_numpy(state))
            action_sequence.append(torch.from_numpy(action))
            intri_reward_sequence.append(intri_reward)
            goal_sequence.append(goal)
            reward_h_sequence.append(reward_h)

            # update the DDPG parameters
            if len(self.replay_buffer[i_level-1].buffer) > self.batch_size:
                self.update(i_level-1, self.n_iter,self.batch_size)
            #experience_buffer_l.add(state, goal, action, intri_reward, next_state, next_goal, done_l)
            # 2.2.5 record segment arguments

            episode_reward_h += reward_h

            if (steps + 1) % self.c == 0 and steps > 0:

                next_goal = self.HAC[i_level].select_action_High(state)

                next_goal = next_goal + np.random.normal(0, self.exploration_state_noise)
                next_goal = next_goal.clip(self.state_clip_low[self.goal_index] , self.state_clip_high[self.goal_index])

                #actor_target_l =  copy.deepcopy(self.HAC[i_level - 1].actor_Low)

                #off-policy correction
                #goal_hat, updated = self.off_policy_correction(actor_target_l, action_sequence, state_sequence, self.goal_dim, self.goal_index, goal_sequence[0], next_state, max_goal, device)
                #print(goal_hat)

                #self.replay_buffer[i_level].add(state, goal_hat, episode_reward_h, next_state, done_h)
                #print("next goal", next_goal)
                #print("goal hat", goal_hat)
                self.replay_buffer[i_level].add((state_sequence[0], next_goal, episode_reward_h, next_state, self.gamma)) # implement goal hat instead of next_goal
                state_sequence, action_sequence, intri_reward_sequence, goal_sequence, reward_h_sequence = [], [], [], [], []
                episode_reward_h = 0
                if len(self.replay_buffer[i_level].buffer) > self.batch_size:
                    self.update(i_level, self.n_iter, self.batch_size)
                #print(episode_reward_h)
                # if state_print_trigger.good2log(t, 500): print_cmd_hint(params=[state_sequence, goal_sequence, action_sequence, intri_reward_sequence, updated, goal_hat, reward_h_sequence], location='training_state')
                # 2.2.9 reset segment arguments & log (reward)




            # 2.2.10 update observations
            state = next_state
            goal = next_goal

            time = time + dt


            goal_achieved = self.check_goal(next_state, final_goal, self.threshold)
            #if goal_achieved and Train == True:
                #print("goal achieved in steps",t)
                #show_goal_achieve = False
                #break
            if goal_achieved and test == True:
                self.solved = True
        


        return next_state
    
    def update(self, k_level, n_iter, batch_size):
    #def update(self, n_iter, batch_size):
        #for i in range(self.k_level):
            if k_level == 0:
                self.HAC[k_level].update_Low(self.replay_buffer[k_level], n_iter, batch_size)
            else:
                self.HAC[k_level].update_High(self.replay_buffer[k_level], n_iter, batch_size)
    
    
    def save(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].save(directory, name+'_level_{}'.format(i))
    
    
    def load(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].load(directory, name+'_level_{}'.format(i))
    
        
        
        
        
        

