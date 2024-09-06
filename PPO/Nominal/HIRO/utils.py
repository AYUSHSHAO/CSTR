import numpy as np
import random
import torch





class ReplayBuffer_Low:
    def __init__(self,max_size = 50000 ): #max_size=5e5
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        assert len(transition) == 7, "transition must have length = 6"
        
        # transiton is tuple of (state, action, reward, next_state, goal, gamma)
        self.buffer.append(transition)
        self.size +=1
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        #print(len(self.buffer))
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, goals, rewards, next_states, next_goals, gamma = [], [], [], [], [], [], []
        
        for i in indexes:
            states.append(np.array(self.buffer[i][0], copy=False))
            actions.append(np.array(self.buffer[i][1], copy=False))
            goals.append(np.array(self.buffer[i][2], copy=False))
            rewards.append(np.array(self.buffer[i][3], copy=False))
            next_states.append(np.array(self.buffer[i][4], copy=False))
            next_goals.append(np.array(self.buffer[i][5], copy=False))
            gamma.append(np.array(self.buffer[i][6], copy=False))

        
        return np.array(states), np.array(actions), np.array(goals), np.array(rewards), np.array(next_states), np.array(next_goals),  np.array(gamma)


class ReplayBuffer_High:
    def __init__(self, max_size = 50000): #max_size=5e5
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0

    def add(self, transition):
        assert len(transition) == 5, "transition must have length = 5"

        # transiton is tuple of (state, action, reward, next_state, goal, gamma)
        self.buffer.append(transition)
        self.size += 1

    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size / 5)]
            self.size = len(self.buffer)
        # print(len(self.buffer))
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        states, goals, rewards, next_states, gamma = [], [], [], [], []

        for i in indexes:
            states.append(np.array(self.buffer[i][0], copy=False))
            goals.append(np.array(self.buffer[i][1], copy=False))
            rewards.append(np.array(self.buffer[i][2], copy=False))
            next_states.append(np.array(self.buffer[i][3], copy=False))
            gamma.append(np.array(self.buffer[i][4], copy=False))

        return np.array(states), np.array(goals), np.array(rewards), np.array(next_states), np.array(gamma)

