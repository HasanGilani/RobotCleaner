import matplotlib.pyplot as plt
import torch
import numpy as np
from CleaningRobotClass import CleaningRobots
from robot_functions_module import (display_room, display_room_permutations)
from EnvClusterManagement import EnvClusterManager
from RandomAgentClass import RandomAgent
from NetworkClass import Network
env_config = {
    'width': 10,
    'max_island_size': 5,
    'min_n_islands': 1,
    'max_n_islands': 5,
    'dirt_fraction': 0.5,
    'n_dirt_generation': False,
    'n_dirt_tiles': 1,
    'seed': 42,
    'max_steps': 1000,        
    'sparse_reward': True,
}

env = CleaningRobots(env_config)

display_room(env.room)

obs = env.observe()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(obs[0], cmap='gray', origin='lower')
axes[0].set_title('Accessible Channel')
axes[1].imshow(obs[1], cmap='gray', origin='lower')
axes[1].set_title('Dirt Channel')
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(10, 2.2))
axes[0].imshow(obs[2], cmap='gray', origin='lower')
axes[0].set_title('Robot Facing Up')
axes[1].imshow(obs[3], cmap='gray', origin='lower')
axes[1].set_title('Robot Facing Right')
axes[2].imshow(obs[4], cmap='gray', origin='lower')
axes[2].set_title('Robot Facing Down')
axes[3].imshow(obs[5], cmap='gray', origin='lower')
axes[3].set_title('Robot Facing Left')
plt.show()


display_room_permutations(env.room)

envs = EnvClusterManager(CleaningRobots, n_envs=8, env_config=env_config)

random_agent = RandomAgent()
random_agent.set_envs(envs)

mean, std = random_agent.evaluate_policy(100)
print(f"Policy mean reward: {mean}, std: {std}")

network_config = {
    'n_filters': 32,   
    'n_blocks': 6,     
    'n_actions': 5,    
    'image_size': 10,  
}

input_tensor = torch.tensor(np.ones((2, 6, 10, 10)), dtype=torch.float32)

network = Network(network_config)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shapes: {network(input_tensor)[0].shape}, {network(input_tensor)[1].shape}")

total_params = sum(p.numel() for p in network.parameters())
print(f"Network parameters: {total_params}")

