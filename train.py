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

max_memory_size = 100000
episodes = 500
dynamics_model_batch_size = 256 

start_time = time.perf_counter()

env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5,
               render_mode="rgb_array")

agent = Agent(env=env)

agent.train(episodes=episodes)

end_time = time.perf_counter()

elapsed_time = end_time - start_time

print(f"Elapsed time was: {elapsed_time}")
        

