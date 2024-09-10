import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor_Low(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, action_bound, action_offset):
        super(Actor_Low, self).__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim, 10)
        self.mu_head = nn.Linear(10, 1)
        self.sigma_head = nn.Linear(10, action_dim)

    def forward(self, state, goal):
        x = torch.cat([state, goal], 1)
        x1 = torch.sigmoid(self.fc1(x))
        mu = 160 * (abs(self.mu_head(x1)))
        #         sigma = (torch.exp((torch.tanh(self.sigma_head(x1)))))
        #         mu = 80*(abs(self.mu_head(x2)))
        #         # sigma = torch.exp((F.tanh(self.sigma_head(x1))))
        sigma = 2 * (abs(self.sigma_head(x1)))
        return Normal(mu, sigma)




class Actor_High(nn.Module):
    def __init__(self, state_dim, goal_dim, goal_index, state_bound, state_offset): # goal_dim as goal is the action taken by the higher level policy
        super(Actor_High, self).__init__()
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, goal_dim),
            nn.Sigmoid()
        )
        # max value of actions
        self.offset = state_offset
        self.bounds = state_bound
        self.offset = state_offset[:,goal_index]
        self.bounds = state_bound[:,goal_index]

    def forward(self, state):
        return self.actor(state)*self.bounds + self.offset



class Critic_Low(nn.Module):
    def __init__(self,state_dim):
        super(Critic_Low, self).__init__()
        self.fc = nn.Linear(state_dim, 10)
        self.v_head = nn.Linear(10, 1)

    def forward(self, state):
        x = F.relu(self.fc(state))
        state_value = self.v_head(x)
        return state_value





class Critic_High(nn.Module):
    def __init__(self, state_dim, goal_dim): # goal_dim as goal is the action taken by the higher level policy
        super(Critic_High, self).__init__()
        # UVFA critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, goal):
        return self.critic(torch.cat([state, goal], 1))


class DDPG_Low:
    def __init__(self, state_dim, action_dim, goal_dim, action_bounds, action_offset, policy_freq, tau, lr):

        ################### For Lower Level Policy############################
        self.tau = tau
        self.policy_freq = policy_freq
        self.goal_dim = goal_dim # as goal is the state which our agent should acheive
        self.actor_Low = Actor_Low(state_dim, action_dim, self.goal_dim, action_bounds, action_offset).to(device)
        self.actor_Low_target = Actor_Low(state_dim, action_dim, self.goal_dim, action_bounds, action_offset).to(device)
        self.actor_optimizer = optim.Adam(self.actor_Low.parameters(), lr=lr)
        # two critics for TD3 learning
        self.critic_Low_1 = Critic_Low(state_dim, action_dim, self.goal_dim).to(device)
        self.critic_Low_target_1 = Critic_Low(state_dim, action_dim, self.goal_dim).to(device)
        self.critic_Low_1_optimizer = optim.Adam(self.critic_Low_1.parameters(), lr=lr)

        self.critic_Low_2 = Critic_Low(state_dim, action_dim, self.goal_dim).to(device)
        self.critic_Low_target_2 = Critic_Low(state_dim, action_dim, self.goal_dim).to(device)
        self.critic_Low_2_optimizer = optim.Adam(self.critic_Low_2.parameters(), lr=lr)

        self.mseLoss = torch.nn.MSELoss()



    def select_action_Low(self, state, goal):
        # print(type(state))
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)

        return self.actor_Low(state, goal).detach().cpu().data.numpy().flatten()

    def update_Low(self, ppo_epochs, batch_obs, batch_log_probs, A_k, batch_rtgs, clip):
        for i in range(ppo_epochs):
            V, curr_log_probs = Evaluate(batch_obs)
            ratios = torch.exp(curr_log_probs - batch_log_probs)
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1.0 - clip, 1.0 + clip) * A_k
            actor_loss = - (torch.min(surr1, surr2).mean())
            # critic_loss = nn.MSELoss()(V, batch_rtgs)
            critic_loss = (batch_rtgs - torch.squeeze(V.T, 0)).pow(2).mean()
            actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_optim.step()
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

    def update_Low(self, buffer, n_iter, batch_size):
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action, goal, reward, next_state, next_goal, gamma = buffer.sample(batch_size)

            # convert np arrays into tensors
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_goal = torch.FloatTensor(next_goal).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size, 1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            goal = torch.FloatTensor(goal).to(device)
            gamma = torch.FloatTensor(gamma).reshape((batch_size, 1)).to(device)

            # select next action
            next_action = self.actor_Low_target(next_state, next_goal).detach()

            # take the minimum of 2 q-values ---> TD3
            target_Q1 = self.critic_Low_target_1(next_state, next_action, next_goal).detach()
            target_Q2 = self.critic_Low_target_2(next_state, next_action, next_goal).detach()
            target_Q = torch.min(target_Q1, target_Q2)
            # Compute target Q-value:
            target_Q = reward + gamma * target_Q

            current_Q1 = self.critic_Low_1(state, action, goal)
            current_Q2 = self.critic_Low_2(state, action, goal)

            # Optimize Critic:
            critic_loss1 = self.mseLoss(current_Q1, target_Q)
            self.critic_Low_1_optimizer.zero_grad()
            critic_loss1.backward()
            self.critic_Low_1_optimizer.step()

            critic_loss2 = self.mseLoss(current_Q2, target_Q)
            self.critic_Low_2_optimizer.zero_grad()
            critic_loss2.backward()
            self.critic_Low_2_optimizer.step()



            if i % self.policy_freq == 0:

                # Compute actor loss:
                actor_loss = -self.critic_Low_1(state, self.actor_Low(state, goal), goal).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor_Low.parameters(), self.actor_Low_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic_Low_1.parameters(), self.critic_Low_target_1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic_Low_2.parameters(), self.critic_Low_target_2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




    def save(self, directory, name):
        torch.save(self.actor_Low.state_dict(), '%s/%s_actor_Low.pth' % (directory, name))
        torch.save(self.critic_Low_1.state_dict(), '%s/%s_critic_Low_1.pth' % (directory, name))
        torch.save(self.critic_Low_2.state_dict(), '%s/%s_critic_Low_2.pth' % (directory, name))


    def load(self, directory, name):
        self.actor_Low.load_state_dict(torch.load('%s/%s_actor_Low.pth' % (directory, name), map_location='cpu'))
        self.critic_Low_1.load_state_dict(torch.load('%s/%s_critic_Low_1.pth' % (directory, name), map_location='cpu'))
        self.critic_Low_2.load_state_dict(torch.load('%s/%s_critic_Low_2.pth' % (directory, name), map_location='cpu'))



class DDPG_High:
    def __init__(self, state_dim, goal_dim, goal_index, state_bound, state_offset, policy_freq, tau, lr):

        ################### For Higher Level_Policy############################
        self.policy_freq = policy_freq
        self.tau = tau
        self.goal_dim = goal_dim  # as goal is the action that higher policy will give
        self.actor_High = Actor_High(state_dim, self.goal_dim, goal_index, state_bound, state_offset).to(device)
        self.actor_High_target = Actor_High(state_dim, self.goal_dim, goal_index, state_bound, state_offset).to(device)
        self.actor_optimizer = optim.Adam(self.actor_High.parameters(), lr=lr)
        # two critics for TD3 learning
        self.critic_High_1 = Critic_High(state_dim, self.goal_dim).to(device)
        self.critic_High_target_1 = Critic_High(state_dim, self.goal_dim).to(device)
        self.critic_High_1_optimizer = optim.Adam(self.critic_High_1.parameters(), lr=lr)

        self.critic_High_target_2 = Critic_High(state_dim, self.goal_dim).to(device)
        self.critic_High_2 = Critic_High(state_dim, self.goal_dim).to(device)
        self.critic_High_2_optimizer = optim.Adam(self.critic_High_2.parameters(), lr=lr)

        self.mseLoss = torch.nn.MSELoss()



    def select_action_High(self, state):
        # print(type(state))
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor_High(state).detach().cpu().data.numpy().flatten()


    def update_High(self, buffer, n_iter, batch_size):
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state,  goal, reward, next_state, gamma = buffer.sample(batch_size)

            # convert np arrays into tensors
            state = torch.FloatTensor(state).to(device)
            goal = torch.FloatTensor(goal).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size, 1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            gamma = torch.FloatTensor(gamma).reshape((batch_size, 1)).to(device)

            # select next action
            next_goal = self.actor_High_target(next_state).detach()

            # take the minimum of 2 q-values ---> TD3
            target_Q1 = self.critic_High_target_1(next_state, next_goal).detach()
            target_Q2 = self.critic_High_target_2(next_state, next_goal).detach()
            target_Q = torch.min(target_Q1, target_Q2)
            # Compute target Q-value:
            target_Q = reward + gamma * target_Q

            current_Q1 = self.critic_High_1(state, goal)
            current_Q2 = self.critic_High_2(state, goal)

            # Optimize Critic:
            critic_loss1 = self.mseLoss(current_Q1, target_Q)
            self.critic_High_1_optimizer.zero_grad()
            critic_loss1.backward()
            self.critic_High_1_optimizer.step()

            critic_loss2 = self.mseLoss(current_Q2, target_Q)
            self.critic_High_2_optimizer.zero_grad()
            critic_loss2.backward()
            self.critic_High_2_optimizer.step()


            if i % self.policy_freq == 0:

                # Compute actor loss:
                actor_loss = -self.critic_High_1(state, self.actor_High(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor_High.parameters(), self.actor_High_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic_High_1.parameters(), self.critic_High_target_1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic_High_2.parameters(), self.critic_High_target_2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, directory, name):

        torch.save(self.actor_High.state_dict(), '%s/%s_actor_High.pth' % (directory, name))
        torch.save(self.critic_High_1.state_dict(), '%s/%s_critic_High_1.pth' % (directory, name))
        torch.save(self.critic_High_2.state_dict(), '%s/%s_critic_High_2.pth' % (directory, name))

    def load(self, directory, name):

        self.actor_High.load_state_dict(torch.load('%s/%s_actor_High.pth' % (directory, name), map_location='cpu'))
        self.critic_High_1.load_state_dict(torch.load('%s/%s_critic_High_1.pth' % (directory, name), map_location='cpu'))
        self.critic_High_2.load_state_dict(torch.load('%s/%s_critic_High_2.pth' % (directory, name), map_location='cpu'))



        ############################################################



class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(5, 10)
        self.v_head = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        state_value = self.v_head(x)
        return state_value

lr = 0.003
ppo_epochs = 10

model1 = ActorNet()
model2 = CriticNet()
actor_optim = optim.Adam(model1.parameters(), lr=lr)
critic_optim = optim.Adam(model2.parameters(), lr=lr)

gamma = 0.99

def compute_rtgs(batch_rews):
    batch_rtgs = []
    for ep_rews in reversed(batch_rews):
        discounted_reward = 0
        for rew in reversed(ep_rews):
            discounted_reward = rew + discounted_reward * gamma
            batch_rtgs.insert(0, discounted_reward)
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

    return batch_rtgs

def Evaluate(batch_obs, batch_act):
    dist = model1(batch_obs)
    V = model2(batch_obs)
    log_probs = dist.log_prob(torch.unsqueeze(batch_acts, 1))
    return V, log_probs.squeeze()

from matplotlib.cbook import safe_first_element
def PPO(ppo_epochs, batch_obs, batch_acts, batch_log_probs, A_k, batch_rtgs):
    for i in range(ppo_epochs):
        V, curr_log_probs = Evaluate(batch_obs, batch_acts)
        ratios = torch.exp(curr_log_probs - batch_log_probs)
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1.0 - clip, 1.0 + clip) * A_k
        actor_loss = - (torch.min(surr1, surr2).mean())
        # critic_loss = nn.MSELoss()(V, batch_rtgs)
        critic_loss = (batch_rtgs - torch.squeeze(V.T, 0)).pow(2).mean()
        actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        LOSS.append(actor_loss.detach().numpy())
        actor_optim.step()
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()
# plt.plot(LOSS)
# plt.show()
