
import numpy as np
from CleaningRobotClass import CleaningRobots
from EnvClusterManagement import EnvClusterManager

class RandomAgent:
    def __init__(self):
        pass
    
    def set_envs(self, envs):
        "Sets vectorised environments for the agent."
        self.n_envs = envs.n_envs
        self.envs = envs
    
    def select_action(self, observations):
        actions = np.array([envs.single_action_space.sample() for _ in range(self.n_envs)])
        return actions
    
    def evaluate_policy(self, n_eval_episodes):
        complete_episode_rewards = []
        accumulated_episode_rewards = np.zeros([self.n_envs])
        observations = self.envs.reset(indices=[*range(self.n_envs)], seeds=[*range(self.n_envs)])
        n_completed_episodes = 0
        
        while n_completed_episodes < n_eval_episodes:
            actions = self.select_action(observations)
            observations, rewards, terminateds, truncateds, _ = self.envs.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            accumulated_episode_rewards += rewards
            
            n_done_environments = np.sum(dones)
            
            if n_done_environments > 0:
                done_episode_indices = np.argwhere(dones)[:, 0]
                new_episode_seeds = [*range(n_completed_episodes + self.n_envs, n_completed_episodes + self.n_envs + n_done_environments)]
                self.envs.reset(indices=done_episode_indices, seeds=new_episode_seeds)
                n_completed_episodes += n_done_environments
                
                completed_episode_rewards = accumulated_episode_rewards[done_episode_indices]
                for r in completed_episode_rewards:
                    complete_episode_rewards.append(r)
                accumulated_episode_rewards[done_episode_indices] = 0
                
        reward_mean = np.mean(complete_episode_rewards)
        reward_std = np.std(complete_episode_rewards)
        return reward_mean, reward_std
    
    
