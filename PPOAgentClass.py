import os
import time
from collections import deque
import torch
import numpy as np
from LoggerClass import Logger
import torch.nn.functional as F
from torch import nn

class PPOAgent:
    def __init__(self, config=None):
        self.config = config if config is not None else self.default_config()
        self.logger = None
        
    def set_envs(self, envs):
        "Sets vectorised environments for the agent."
        assert hasattr(envs, 'n_envs'), "This PPO agent requires use of a vectorised environment wrapper."
        self.envs = envs
    
    def train(self):
        "Trains the agent for a specified number of timesteps."
        assert hasattr(self, 'network'), "Call `set_network` method before trying to train model."
              
        buffer_states   = torch.zeros((self.config['n_rollout_steps'], self.config['n_envs']) + self.envs.single_observation_space.shape).to(self.config['device'])
        buffer_actions  = torch.zeros((self.config['n_rollout_steps'], self.config['n_envs'])).to(self.config['device'])
        buffer_rewards  = torch.zeros((self.config['n_rollout_steps'], self.config['n_envs'])).to(self.config['device'])
        buffer_dones    = torch.zeros((self.config['n_rollout_steps'], self.config['n_envs'])).to(self.config['device'])
        buffer_logprobs = torch.zeros((self.config['n_rollout_steps'], self.config['n_envs'])).to(self.config['device'])
        buffer_values   = torch.zeros((self.config['n_rollout_steps'], self.config['n_envs'])).to(self.config['device'])
        
        s_time = time.time()
        current_training_step = 0
        
        states = torch.tensor(self.envs.reset(), dtype=torch.float32).to(self.config['device'])
        dones = torch.zeros(self.config['n_envs']).to(self.config['device'])
        
        episodic_return = np.zeros(self.config['n_envs'])
        episodic_step_count = np.zeros(self.config['n_envs'])
        rewards_deque = deque([0.], maxlen=20)  
        episode_duration_deque = deque(maxlen=20)  
        
        for epoch in range(1, self.config['n_epochs'] + 1):
            
            print(f"\rEpoch {epoch}/{self.config['n_epochs']}, mean reward: {np.mean(rewards_deque):.3f}", end="")
            
            if self.config['anneal_lr']:
                fraction = 1.0 - ((epoch - 1.0) / self.config['n_epochs'])
                lr_current = fraction * self.config['learning_rate']
                self.optimizer.param_groups[0]['lr'] = lr_current
                
            for step in range(0, self.config['n_rollout_steps']):
                current_training_step += self.config['n_envs']
                
                buffer_states[step] = states
                buffer_dones[step]  = dones
                
                with torch.no_grad():
                    enc = self.network.base(states)                              
                    logits = self.network.policy_head(enc)                       
                    dist = torch.distributions.Categorical(logits=logits)        
                    actions = dist.sample()                                      
                    buffer_actions[step] = actions 
                    buffer_logprobs[step] = dist.log_prob(buffer_actions[step])  
                    buffer_values[step]   = self.network.value_head(enc).view(-1)     
        
                states, rewards, terminateds, truncateds, infos = self.envs.step(actions.cpu().numpy())
                dones = np.logical_or(terminateds, truncateds)
                buffer_rewards[step] = torch.tensor(rewards, dtype=torch.float32).to(self.config['device'])
                
                
                episodic_return += rewards
                episodic_step_count += 1
                
                if True in dones:
                    for env_idx, done_flag in enumerate(dones):    
                        if done_flag:
                            rewards_deque.append(episodic_return[env_idx])
                            episode_duration_deque.append(episodic_step_count[env_idx])
                            episodic_return[env_idx], episodic_step_count[env_idx] = 0., 0.
                    n_completed_episodes = sum(dones)
                    done_episode_indices = np.argwhere(dones)[:, 0]
                    seeds = [self.config['seed'] for _ in range(n_completed_episodes)]
                    self.envs.reset(indices=done_episode_indices, seeds=seeds)
                    states = self.envs.observe()
                    
                states = torch.tensor(states, dtype=torch.float32).to(self.config['device']) 
                dones = torch.tensor(dones, dtype=torch.float32).to(self.config['device'])
                
                
            with torch.no_grad():
                enc = self.network.base(states)
                next_value = self.network.value_head(enc).view(-1)

                advantages = torch.zeros_like(buffer_rewards).to(self.config['device'])

                lastgaelam = 0

                for t in reversed(range(self.config['n_rollout_steps'])):
                    if t == self.config['n_rollout_steps'] - 1:        
                        next_non_terminal = 1.0 - buffer_dones[t]
                        next_values = next_value
                    else:
                        next_non_terminal = 1.0 - buffer_dones[t + 1]  
                        next_values = buffer_values[t + 1]

                    delta = buffer_rewards[t] + self.config['gamma'] * next_values * next_non_terminal - buffer_values[t]

                    advantages[t] = lastgaelam = delta + self.config['gamma'] * self.config['gae_lambda'] * next_non_terminal * lastgaelam

                returns = advantages + buffer_values
                
            clipfracs = []
            approx_kls = []

            batch_states     = buffer_states.reshape((-1,) + self.envs.single_observation_space.shape)
            batch_actions    = buffer_actions.reshape((-1,) + self.envs.single_action_space.shape)
            batch_logprobs   = buffer_logprobs.reshape(-1)
            batch_advantages = advantages.reshape(-1)
            batch_returns    = returns.reshape(-1)
            batch_values     = buffer_values.reshape(-1)
            
            batch_indices = np.arange(self.config['batch_size'])
            for learning_update_idx in range(self.config['n_learning_updates_per_batch']):
                np.random.shuffle(batch_indices)
                for start in range(0, self.config['batch_size'], self.config['minibatch_size']):
                    end = start + self.config['minibatch_size']
                    mb_indices = batch_indices[start:end]
                    
                    new_logits, new_values = self.network(batch_states[mb_indices])  
                    new_values = new_values.view(-1)
                    dist = torch.distributions.Categorical(logits=new_logits)  
                    new_logprobs = dist.log_prob(batch_actions[mb_indices])    
                    entropies = dist.entropy()                                 
                    
                    logratio = new_logprobs - batch_logprobs[mb_indices]       
                    ratio = logratio.exp()                                     
                    
                    with torch.no_grad():
                        approx_kls += [((ratio - 1) - logratio).mean()]  
                        clipfracs  += [((ratio - 1.).abs() > self.config['clip_coef']).float().mean()]  
                    
                    mb_advantages = batch_advantages[mb_indices]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                    loss_surrogate_unclipped = -mb_advantages * ratio
                    loss_surrogate_clipped   = -mb_advantages * torch.clip(ratio,                         
                                                                           1 - self.config['clip_coef'],  
                                                                           1 + self.config['clip_coef'])  
                    loss_policy = torch.max(loss_surrogate_unclipped, loss_surrogate_clipped).mean()
                    
                    loss_v_unclipped = (new_values - batch_returns[mb_indices]) ** 2  
                    value_clipped = batch_values[mb_indices] + torch.clip(new_values - batch_returns[mb_indices], 
                                                                          -self.config['clip_coef'],              
                                                                          self.config['clip_coef'])               
                    
                    loss_v_clipped = (value_clipped - batch_returns[mb_indices]) ** 2  
                    loss_v_max = torch.max(loss_v_unclipped, loss_v_clipped)
                    loss_value = 0.5 * loss_v_max.mean()                               
                    
                    loss_entropy = entropies.mean()
                    
                    loss = loss_policy + \
                           self.config['value_loss_weight'] * loss_value + \
                           -self.config['entropy_loss_weight'] * loss_entropy
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.config['max_grad_norm'])
                    self.optimizer.step()
                    
            steps_per_second = int(current_training_step / (time.time() - s_time))
            mean_clipfrac = np.mean([item.cpu().numpy() for item in clipfracs])
            mean_kl_div = np.mean([item.cpu().numpy() for item in approx_kls])
            
            y_pred, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            
            epoch_logs = {
                'mean_training_episode_reward': np.mean(rewards_deque),
                'training_episode_reward_var': np.var(rewards_deque),
                'mean_episode_duration': np.mean(episode_duration_deque),
                'policy_loss': loss_policy.item(),
                'value_loss': loss_value.item(),
                'entropy_loss': loss_entropy.item(),
                'mean_kl_div': mean_kl_div,
                'clipfrac': mean_clipfrac,
                'explained_variance': explained_var,
                'learning_rate': lr_current,
                'steps_per_second': steps_per_second,
            }
            
            self.log(epoch_logs)
    
    def log(self, logs_dict):
        "Logs training information to file which can optionally be saved to disk."
        if self.logger == None:
            self.logger = Logger(logs_filename=self.config['run_name'] + '-logs')
            self.logger.delete_logs(verbose=False)
        self.logger.log(logs_dict)
        
    def select_action(self, states, deterministic=False):

        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.config['device'])

        enc = self.network.base(states_tensor)
        logits = self.network.policy_head(enc)

        if deterministic:
            actions = torch.argmax(logits, axis=1).cpu().numpy()
        else:
            actions = torch.distributions.Categorical(logits=logits).sample().detach().cpu().numpy()

        return actions

    def action_probability(self, states, return_logits=False):
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.config['device'])

        enc = self.network.base(states_tensor)
        logits = self.network.policy_head(enc)

        if return_logits:
            return logits
        else:
            return torch.distributions.Categorical(logits=logits).probs.detach().cpu().numpy()
    
    def evaluate_policy(self, n_eval_episodes):
        complete_episode_rewards = []
        accumulated_episode_rewards = np.zeros(self.config['n_envs'])
        observations = self.envs.reset(indices=[*range(self.config['n_envs'])], seeds=[*range(self.config['n_envs'])])
        n_completed_episodes = 0
        
        while n_completed_episodes < n_eval_episodes:
            actions = self.select_action(observations, deterministic=True)
            observations, rewards, terminateds, truncateds, _ = self.envs.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            accumulated_episode_rewards += rewards
            
            n_done_environments = np.sum(dones)
            
            if n_done_environments > 0:
                done_episode_indices = np.argwhere(dones)[:, 0]
                new_episode_seeds = [*range(n_completed_episodes + self.config['n_envs'], n_completed_episodes + self.config['n_envs'] + n_done_environments)]
                self.envs.reset(indices=done_episode_indices, seeds=new_episode_seeds)
                n_completed_episodes += n_done_environments
                
                completed_episode_rewards = accumulated_episode_rewards[done_episode_indices]
                for r in completed_episode_rewards:
                    complete_episode_rewards.append(r)
                accumulated_episode_rewards[done_episode_indices] = 0
                
        reward_mean = np.mean(complete_episode_rewards)
        reward_std = np.std(complete_episode_rewards)
        return reward_mean, reward_std
    
    def set_network(self, network):
        "Sets user defined policy and value networks after validating network is valid."
        required_attributes = ['base', 'policy_head', 'value_head']
        for attr in required_attributes:
            assert hasattr(network, attr), f"Network is missing required property: {attr}"
        assert hasattr(network, 'forward'), "Network is missing the required forward method."
        
        self.network = network
        
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.config['learning_rate'], eps=1e-5)  
    
    def reset(self):
        "Reinitialises network parameters and resets environment(s)"
        if hasattr(self, 'network'):
            for layer in self.network.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        if hasattr(self, 'envs'):
            self.envs.reset()
    
    def save(self, path=None):
        "Saves model and config to specified path."
        assert hasattr(self, 'network'), "Network not found."
        if path is None:
            directory = 'saved-models'
            if not os.path.exists(directory):
                os.makedirs(directory)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"ppo-agent-{timestamp}.pt"
            path = os.path.join(directory, filename)
        
        save_dict = {
            'network_state_dict': self.network.state_dict(),
            'config': self.config
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        "Loads model and config from specified path."
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.config = checkpoint['config']
        print(f"Model loaded from {path}")
    
    def default_config(self):
        config = {
            
            'run_name': 'cleaning-robots-experiment-1',                
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',  
            'create_logs': True,                                       

            
            'n_envs': 8,                        
            'training_steps': 100000,           
            'n_rollout_steps': 64,              
            'n_minibatches': 4,                 
            'n_learning_updates_per_batch': 5, 

            
            'learning_rate': 2.5e-4,      
            'anneal_lr': True,            
            'gamma': 0.99,                
            'clip_coef': 0.1,             
            'gae_lambda': 0.95,           
            'value_loss_weight': 0.5,     
            'entropy_loss_weight': 0.01,  
            'max_grad_norm': 0.5,         
        }

        config['batch_size']     = config['n_envs'] * config['n_rollout_steps']
        config['minibatch_size'] = config['batch_size'] // config['n_minibatches']
        config['n_epochs']       = config['training_steps'] // config['batch_size']
        
        return config

