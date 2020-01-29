import random
from collections import deque
import numpy as np

import torch


class ReplayBuffer(object):
    def __init__(self, config):
        self.config = config

        self.capacity = self.config.memory_capacity
        self.batch_size = self.config.batch_size
        self.memory = deque(maxlen=config.memory_capacity)
        self.position = 0

    def length(self):
        return len(self.memory)

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state[0], action, reward, next_state[0], done))

    def sample_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states = torch.stack([elem[0] for elem in batch]).float()
        actions = torch.from_numpy(np.asarray([elem[1] for elem in batch])).long().unsqueeze(1)
        rewards = torch.from_numpy(np.asarray([elem[2] for elem in batch])).float().unsqueeze(1)
        next_states = torch.stack([elem[3] for elem in batch]).float()
        dones = torch.from_numpy(np.asarray([elem[4] for elem in batch])).float().unsqueeze(1)

        return states, actions, rewards, next_states, dones