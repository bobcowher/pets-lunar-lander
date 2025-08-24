import gymnasium as gym
import time
import torch
from torch._prims_common import device_or_default
from torch.nn.functional import mse_loss
from buffer import ReplayBuffer
from model import DynamicsModel 
from torch.utils.tensorboard import SummaryWriter
import datetime
from agent import Agent

max_memory_size = 10000
episodes = 1000
dynamics_model_batch_size = 256 

env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5,
               render_mode="rgb_array")

agent = Agent(env=env)

agent.train()
        

