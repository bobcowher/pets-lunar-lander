from os import stat
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsModel(nn.Module):

    def __init__(self, hidden_dim=256, obs_shape=None, action_shape=None) -> None:
        super(DynamicsModel, self).__init__()

        self.fc1 = nn.Linear(obs_shape + action_shape, hidden_dim) # Accept the state concatenated with reward.
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.obs_diff_output = nn.Linear(hidden_dim, obs_shape) # Return the state dimensions + the reward
        self.reward_output = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=-1)
        print(f" X Shape: {x.shape}")
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        obs_diff = self.obs_diff_output(x)
        reward = self.reward_output(x)

        return obs_diff, reward
        


