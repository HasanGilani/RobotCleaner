import torch
from EnvClusterManagement import EnvClusterManager
from PPOAgentClass import PPOAgent
from NetworkClass import Network
from CleaningRobotClass import CleaningRobots
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.animation as animation
env_config = {
    'width': 10,
    'max_island_size': 5,
    'min_n_islands': 1,
    'max_n_islands': 5,
    'dirt_fraction': 0.5,
    'n_dirt_generation': True,
    'n_dirt_tiles': 1,
    'seed': 42,
    'max_steps': 50,        
    'sparse_reward': True,
}
checkpoint = torch.load('ppo_agent.pth')
envs = EnvClusterManager(CleaningRobots, n_envs=1, env_config=env_config)
done = False

network = Network(checkpoint['network_config']).to(checkpoint['ppo_config']['device'])
network.load_state_dict(checkpoint['network_state_dict'])
agent = PPOAgent(checkpoint['ppo_config'])
agent.set_envs(envs)
agent.set_network(network)
agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


while not done:
    actions = agent.select_action(envs.observe(), deterministic=True)
    states, rewards, terminateds, truncateds, infos = envs.step(actions)
    done = True in np.logical_or(terminateds, truncateds)
    
print('done')


animation = envs.render()
print(animation)
HTML(animation.to_jshtml())


animation.save('animation.gif', writer='imagemagick', fps=30)

plt.show()
