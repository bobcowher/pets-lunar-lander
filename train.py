import gymnasium as gym
import time
import torch
from torch._prims_common import device_or_default
from torch.nn.functional import mse_loss
from buffer import ReplayBuffer
from model import DynamicsModel 

max_memory_size = 10000
episodes = 100

env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5,
               render_mode="rgb_array")

obs, info = env.reset()
action = env.action_space.sample()

print(obs.shape)
print(action.shape)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

memory = ReplayBuffer(max_size=max_memory_size, 
                      input_shape=obs.shape,
                      n_actions=action.shape[0],
                      input_device=device,
                      output_device=device)
        
dynamics_model = DynamicsModel(obs_shape=obs.shape[0], action_shape=action.shape[0])
dynamics_model = dynamics_model.to(device)
optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=0.0001)

# def __init__(self, max_size, input_shape, n_actions,
#                  input_device, output_device='cpu', frame_stack=4):
#
for episode in range(episodes):
    done = False
    episode_reward = 0
    obs, info = env.reset()

    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        memory.store_transition(obs, action, reward, next_obs, done)
        obs = next_obs
        episode_reward = episode_reward + reward
        env.render()
        if(done):
            break

    if(memory.can_sample(batch_size=32)):
        states, actions, rewards, next_states, dones = memory.sample_buffer(batch_size=32)
       
        # actions = actions.unsqueeze(1).long()
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1).float()

        predicted_obs_diffs, predicated_rewards = dynamics_model(states, actions)

        loss = mse_loss(next_states - states, predicted_obs_diffs) + mse_loss(rewards, predicated_rewards)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    print(f"Episode {episode} finished. Reward: {episode_reward}")

