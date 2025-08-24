from os import stat
import os
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
        self.model_save_dir = 'models'

    def forward(self, state, action):

        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        obs_diff = self.obs_diff_output(x)
        reward = self.reward_output(x)

        return obs_diff, reward
        
    def save_the_model(self, filename='latest.pt'):
        os.makedirs(self.model_save_dir, exist_ok=True) 
        torch.save(self.state_dict(), f"{self.model_save_dir}/{filename}")

    def load_the_model(self, filename='latest.pt'):
        try:
            self.load_state_dict(torch.load(f"{self.model_save_dir}/{filename}"))
            print(f"Loaded weights from {self.model_save_dir}/{filename}")
        except FileNotFoundError:
            print(f"No weights file found at {self.model_save_dir}/{filename}")

