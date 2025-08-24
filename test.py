import gymnasium as gym
from agent import Agent

max_memory_size = 10000
episodes = 1000
dynamics_model_batch_size = 256 

env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5,
               render_mode="human")

agent = Agent(env=env)

agent.test()
        

