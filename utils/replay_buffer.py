from collections import deque
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)

        # Convert all components to tensors
        states = torch.stack([torch.tensor(state, dtype=torch.float32) for state in states])
        actions = torch.stack([torch.tensor(action, dtype=torch.int64) for action in actions])
        rewards = torch.stack([torch.tensor(reward, dtype=torch.float32) for reward in rewards])
        next_states = torch.stack([torch.tensor(next_state, dtype=torch.float32) for next_state in next_states])
        dones = torch.stack([torch.tensor(done, dtype=torch.float32) for done in dones])

        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)
