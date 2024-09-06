
import math
from math import exp
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#######################################################

def get_state(action,tf,x0): # action = [21, 34, ....]
    F = action
    t_model = np.round_(np.linspace(0, tf,15), decimals = 8)
    def model(y,t):
        #y = x0
    #         F = 0.0058                               #kg/sec
        k1=(3.92E7/60*exp(-6614.83/y[6]))
        k2=(5.77E5/60*exp(-(4997.98)/y[6]))
        k3=(5.88E12/60*exp(-(9993.96)/y[6]))
        k4=(0.98E10/60*exp(-7366.64/y[6]))
        k5=(5.35E3/60*exp(-(3231.18)/y[6]))
        k6=(2.15E4/60*exp(-4824.87/y[6]))

        V = 1             #m^3
        rho = 860         #kg/m^3
        Mr = 391.4        #Kg/Kmol
        Cm = 1277         #kJ/(kmol K)
        Delta_H = -18500  #kj/kmol
        AU = 7.5          #kj/(sec K) = 450 kj/(min K)
        mj= 99.69         #kg
        Cw = 4.2          #kj/(kg K)
        Tc = 293.15       #K


        y1 = y[0]
        y2 = y[1]
        y3 = y[2]
        y4 = y[3]
        y5 = y[4]
        y6 = y[5]
        y7 = y[6]
        y8 = y[7]


        dy1dt = -k1*y1*y5+k2*y2*y4                                                    # TG
        dy2dt = k1*y1*y5-k2*y2*y4-k3*y2*y5+k4*y3*y4                                   # DG
        dy3dt = -k5*y3*y5+k3*y2*y5+k6*y6*y4-k4*y3*y4                                  #MG
        dy4dt = k1*y1*y5-k2*y2*y4+k3*y2*y5-k4*y3*y4+k5*y3*y5-k6*y6*y4                 #ME
        dy5dt = -dy4dt                                                                #A
        dy6dt = k5*y3*y5-k6*y6*y4                                                     #Gl
        dy7dt = (Mr/(V*rho*Cm))*((-dy4dt*V*Delta_H)+(AU*(y8-y7)))                     #Tr                                                               #Tr
        dy8dt =  ((F*(Tc-y8))-((AU/Cw)*(y8-y7)))/mj                                   #Tj


        return [dy1dt, dy2dt, dy3dt, dy4dt, dy5dt, dy6dt, dy7dt, dy8dt]

    t = np.linspace(0, tf,10)
    y = odeint(model, x0,t)
    All=y

    return All


x0=[0.347, 0, 0, 0, 2.082, 0, 338, 300]
state = get_state(0,6000,x0)
print(state)


plt.figure()
plt.plot([l for i,j,k,l,m,n,o,p in state])
plt.title("FAME")
plt.show()


import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.max_action * torch.tanh(self.layer3(x))
        return x

# Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = torch.relu(self.layer1(torch.cat([x, u], 1)))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# TD3 agent
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
