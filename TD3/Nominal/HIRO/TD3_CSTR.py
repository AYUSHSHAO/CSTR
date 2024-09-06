import numpy as np
from collections import deque
import random
import math
from math import exp
from scipy.integrate import odeint
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable
import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from gym import spaces
tanh = np.tanh
pow = math.pow
#exp = np.exp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


seed = 12369
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)



global dt
dt = 0.05


def CSTR(action, ti, x0):
    Fa0 = action

    def model(y, ti):
        x1 = y[0]  # Propylene oxide
        x2 = y[1]  # water with H2SO4
        x3 = y[2]  # Propylene glycol
        x4 = y[3]  # Methanol
        x5 = y[4]  # Temperature
        parameters = [1000, 75, 16000, 60, 100, 1000]
        Fb0, T0, UA, Ta1, Fm0, mc = parameters
        V = (1 / 7.484) * 500
        k = 16.96e12 * math.exp(-32400 / 1.987 / (y[4] + 460))
        # ra = -k*Ca
        # rb = -k*Ca
        # rc = k*Ca
        Nm = y[3] * V
        Na = y[0] * V
        Nb = y[1] * V
        Nc = y[2] * V

        ThetaCp = 35 + Fb0 / Fa0[0] * 18 + Fm0 / Fa0[0] * 19.5
        v0 = Fa0[0] / 0.923 + Fb0 / 3.45 + Fm0 / 1.54
        Ta2 = y[4] - (y[4] - Ta1) * exp(-UA / (18 * mc))
        Ca0 = Fa0[0] / v0
        Cb0 = Fb0 / v0
        Cm0 = Fm0 / v0
        Q = mc * 18 * (Ta1 - Ta2)
        tau = V / v0
        NCp = Na * 35 + Nb * 18 + Nc * 46 + Nm * 19.5

        dx1_dt = 1 / tau * (Ca0 - x1) - k * x1
        dx2_dt = 1 / tau * (Cb0 - x2) - k * x1
        dx3_dt = 1 / tau * (0 - x3) + k * x1
        dx4_dt = 1 / tau * (Cm0 - x4)
        dx5_dt = (Q - Fa0[0] * ThetaCp * (x5 - T0) + (-36000) * (-k * x1) * V) / NCp

        return np.array([dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt])

    t = np.linspace(ti, ti + dt, 100)
    y = odeint(model, x0, t)
    All = y[-1]
    p = All[2]
    A = y[99, 0]
    B = y[99, 1]
    C = y[99, 2]
    D = y[99, 3]
    E = y[99, 4]
    rewards = -np.abs((All[2] - 0.143))

    return All, rewards



class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state):
        experience = (state, action, reward, next_state)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            state, action, reward, next_state = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)

        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.buffer)



class Actor(nn.Module):
    def __init__(self, hidden_size, learning_rate=3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(5, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        #         x = self.linear3(x)
        return x


class Critic(nn.Module):
    def __init__(self, hidden_size):
        super(Critic, self).__init__()
        # Q1
        self.linear1 = nn.Linear(6, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        # Q2
        self.linear4 = nn.Linear(6, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state.reshape((5, 5)), action.reshape((5, 1))], dim=1)
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        x2 = F.relu(self.linear4(x))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2

    def get_Q(self, state, action):
        x = torch.cat([state.reshape((5, 5)), action.reshape((5, 1))], dim=1)
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1


def norm_action(action):
    low = -1
    high = 1

    action = ((action - low) / (high - low))

    action = 78 + action

    return action


class TD3(object):
    def __init__(self, action, states, max_action, min_action, num_actions, hidden_size=256, actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, policy_freq=2, policy_noise=0.2, noise_clip=0.5,
                 max_memory_size=50000):
        self.num_states = states
        self.num_actions = action
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.max_action = max_action
        self.min_action = min_action

        self.actor = Actor(hidden_size).to(device)
        self.actor_target = Actor(hidden_size).to(device)

        self.critic = Critic(hidden_size).to(device)
        self.critic_target = Critic(hidden_size).to(device)

        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state, noise=1):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(device)
        action = self.actor.forward(state).to(device)
        action = action.cpu().detach().numpy()
        action = (action + np.random.normal(1, noise, size=self.num_actions))
        action.clip(self.min_action, self.max_action)
        return action.clip(self.min_action, self.max_action)

    def train(self, iterations, batch_size, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=4):
        states, actions, rewards, next_states = self.memory.sample(batch_size)
        states = torch.transpose(Variable(torch.from_numpy(np.array(states)).float().unsqueeze(0)), 0, 1).to(device)
        actions = (Variable(torch.from_numpy(np.array(actions)).float())).to(device)
        rewards = Variable(torch.from_numpy(np.array(rewards)).float().unsqueeze(1)).to(device)
        next_states = torch.transpose(Variable(torch.from_numpy(np.array(next_states)).float().unsqueeze(0)), 0, 1).to(
            device)

        for it in range(iterations):
            noise = actions.data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_actions = self.actor_target.forward(next_states)
            #next_actions = norm_action(next_actions)
            next_actions = (next_actions + noise).clamp(self.min_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target.forward(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (discount * target_Q).detach()
            current_Q1, current_Q2 = self.critic.forward(states, actions)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            if it % policy_freq == 0:
                actor_loss = -self.critic.get_Q(states, self.actor(states)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))



def plot_G(propylene_glycol, tot_time, flowrate, name):
    time = np.linspace(0, tot_time, int(tot_time / dt))
    T1 = 0.143  # target
    ta = np.ones(int(tot_time / dt)) * T1
    fig, ax1 = plt.subplots()
    # time = np.linspace(0,T,int(T/dt))
    font1 = {'family': 'serif', 'size': 15}
    font2 = {'family': 'serif', 'size': 15}
    color = 'tab:red'
    ax1.set_xlabel('time (min)', fontdict=font1)
    ax1.set_ylabel('Propylene Glycol', fontdict=font2, color=color)
    ax1.plot(time, propylene_glycol, color=color)
    ax1.plot(time, ta, color='tab:orange', linewidth=4, label='reference concentration')
    leg = ax1.legend(loc='lower right')

    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('flowrate', fontdict=font2, color=color)  # we already handled the x-label with ax1
    ax2.step(time, flowrate, where='post', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(color='g', linestyle='-', linewidth=1)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.savefig('deeprl_test1_batch1.jpg')
    plt.savefig(name + '.png')
    plt.close()





#seed = 50
#torch.manual_seed(seed)
high = np.array([10, 10, 10, 10, 200])

observation_space = spaces.Box(
    low=np.array([0, 3.45, 0, 0, 75]),
    high=high,
    dtype=np.float32
)
high = np.array([80], dtype=np.float32)

action_space = spaces.Box(
    low=np.array([0]),
    high=high,
    dtype=np.float32
)
print("action shape shape",action_space)
agent = TD3(action_space.shape[0], observation_space.shape[0], max_action=action_space.high[0],
            min_action=action_space.low[0], num_actions=action_space.shape[0])



batch_size = 5
tot_time = 4
rewards = []
avg_rewards = []
rmse = []
avg_rmse = []
loss_p = []
loss_c = []
episode_reward = []
CSTR_TD3 = []
IAE = []
avg_IAE = []

directory_TD3 = "./TD3/Reward_Plots/"
directory_TD3_plot_G = "./TD3/Plot_G/"



def check_goal(state, goal=0.143, threshold=0.001):
    difference = abs(state[2] - goal)

    if difference > threshold:
        return False
    return True

starting_state = [0, 3.45, 0, 0, 75]


for episode in range(500):
    goal_concentration_reached = False
    timetaken = 0
    propylene_glycol = []
    flowrate = []
    x0 = starting_state
    t = 0.01
    last_state = x0
    viability = []
    episode_reward = 0
    total_reward = 0

    lo = 0  # rmse
    iae = 0
    while t < tot_time:

        propylene_glycol.append(last_state[2])  # output variable
          # concentration
        action = agent.get_action(np.array(last_state))

        action_env = norm_action(action[0])
        action_env = action_env.clip(action_space.low, action_space.high)
        #         flowrate.append(action[0])
        flowrate.append(action_env[0])
        #         new_state, reward = CSTR(action[0],t, x0)
        new_state, reward = CSTR(action_env, t, last_state)
        goal_acheived = check_goal(new_state)
        #if goal_acheived:
            #print("################ Solved! ################ ")

            #break
        agent.memory.push(x0, [action_env], reward, new_state)
        if len(agent.memory) > batch_size:
            agent.train(6, batch_size)
        t = t + dt
        last_state = new_state
        if last_state[2] >= 0.143 and not goal_concentration_reached:
            goal_concentration_reached = True
            timetaken = last_state[2]
        lo += (np.abs(new_state[2] - 0.143) ** 2)
        iae += (np.abs(new_state[2] - 0.143))
        reward = np.array(reward).flatten()
        episode_reward += reward

    #if episode == 2:
        #plot_G(propylene_glycol, flowrate)
    name = directory_TD3_plot_G+ str(episode+1)

    lo = math.sqrt(lo / 40)
    IAE.append(iae)
    rmse.append(lo)
    rewards.append(episode_reward[0])
    avg_rewards.append(np.mean(rewards[-10:]))
    avg_rmse.append(np.mean(rmse[-10:]))
    avg_IAE.append(np.mean(IAE[-10:]))
    CSTR_TD3.append(propylene_glycol)
    plot_G(propylene_glycol, tot_time, flowrate, name)
    print("batch: ", episode + 1, " reward: ", episode_reward)
    print("last state", last_state[2])
    print("Time taken: ", timetaken)
    print("RMSE:", rmse[episode])


np.savetxt("CSTR_TD3.csv", CSTR_TD3, delimiter=",")

# In[11]:

font1 = {'family': 'serif', 'size': 15}
font2 = {'family': 'serif', 'size': 15}

plt.figure()
plt.plot(rewards)
plt.xlabel("Number of episodes", fontdict=font2)
plt.ylabel("Rewards", fontdict=font2)
plt.savefig(directory_TD3+'Reward_Per_Episode_TD3.png', bbox_inches = 'tight')
plt.close()

#plt.show()

plt.figure()
plt.plot(avg_rewards)
plt.xlabel("Number of episodes", fontdict=font1)
plt.ylabel("Average Rewards", fontdict=font2)
plt.savefig(directory_TD3+'Average_Rewarde_TD3.png', bbox_inches = 'tight')
plt.close()

#plt.show()
# In[ ]:
plt.figure()
plt.plot(avg_rmse)
plt.xlabel("Number of episodes", fontdict=font2)
plt.ylabel("Average RMSE", fontdict=font2)
plt.savefig(directory_TD3+'RMSE_TD3.png', bbox_inches = 'tight')
plt.close()

#plt.show()

plt.figure()
plt.plot(avg_IAE)
plt.xlabel("Number of episodes", fontdict=font2)
plt.ylabel("Average IAE", fontdict=font2)
plt.savefig(directory_TD3+'IAE_TD3.png', bbox_inches = 'tight')
plt.close()

