import torch
import torch.nn as nn
import torch.optim as optim

class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
