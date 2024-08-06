import torch
from EnvClusterManagement import EnvClusterManager
from PPOAgentClass import PPOAgent
from NetworkClass import Network
from CleaningRobotClass import CleaningRobots

env_config = {
    'width': 10,                  
    'max_island_size': 5,         
    'min_n_islands': 1,           
    'max_n_islands': 5,           
    'dirt_fraction': 0.5,         
    'n_dirt_generation': True,    
    'n_dirt_tiles': 1,            
    'seed': None,                 
    'max_steps': 50,            
    'sparse_reward': True,        
}

network_config = {
    'n_filters': 32,              
    'n_blocks': 6,                
    'n_actions': 5,               
    'image_size': 10,             
}

ppo_config = {
    'run_name': 'cleaning-robots-experiment-1',                
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  
    'create_logs': True,                                       
    
    'n_envs': 32,                        
    'training_steps': 1000000,           
    'n_rollout_steps': 32,              
    'n_minibatches': 8,                 
    'n_learning_updates_per_batch': 3,  
    
    'learning_rate': 5e-4,      
    'anneal_lr': True,            
    'gamma': 0.99,                
    'clip_coef': 0.1,             
    'gae_lambda': 0.95,           
    'value_loss_weight': 0.5,     
    'entropy_loss_weight': 0.05,  
    'max_grad_norm': 0.5,         
    'seed': None,
}

ppo_config['batch_size']     = ppo_config['n_envs'] * ppo_config['n_rollout_steps']
ppo_config['minibatch_size'] = ppo_config['batch_size'] // ppo_config['n_minibatches']
ppo_config['n_epochs']       = ppo_config['training_steps'] // ppo_config['batch_size']

print(f"Batch size: {ppo_config['batch_size']}")
print(f"Minibatch size: {ppo_config['minibatch_size']}")

def validate_ppo_config(ppo_config):
    "Catches any potential issues with user-defined config"
    assert type(ppo_config['batch_size']) == int, "Batch size must be integer."
    assert ppo_config['batch_size'] % ppo_config['n_minibatches'] == 0, "Batch size must be divisible by minibatch size."
    print("\nConfig file validated.\n")

validate_ppo_config(ppo_config)

envs = EnvClusterManager(CleaningRobots, n_envs=ppo_config['n_envs'], env_config=env_config)
ntwrk = Network(network_config).to(ppo_config['device'])
agent = PPOAgent(ppo_config)

agent.set_envs(envs)
agent.set_network(ntwrk)

agent.train()

torch.save({
    'network_state_dict': agent.network.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
    'ppo_config': ppo_config,
    'network_config': network_config
}, 'ppo_agent.pth')
