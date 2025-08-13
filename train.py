import gymnasium as gym
import time
from buffer import ReplayBuffer

max_memory_size = 10000

env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5,
               render_mode="human")

obs, info = env.reset()
action = env.action_space.sample()

print(obs.shape)
print(action.shape)

memory = ReplayBuffer(max_size=max_memory_size, 
                      input_shape=obs.shape,
                      n_actions=action.shape[0],
                      input_device="cpu",
                      output_device="gpu")
        

# def __init__(self, max_size, input_shape, n_actions,
#                  input_device, output_device='cpu', frame_stack=4):
#
for _ in range(1000):
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)
    memory.store_transition(obs, action, reward, next_obs, done)
    obs = next_obs
    env.render()
    if(done):
        break

    if(memory.can_sample(batch_size=32)):
        states, actions, rewards, next_states, dones = memory.sample_buffer(batch_size=32)
        print(states)

