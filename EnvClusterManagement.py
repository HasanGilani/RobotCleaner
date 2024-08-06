import numpy as np

class EnvClusterManager:
   
    def __init__(self, env, n_envs, env_config=None):
        self.n_envs = n_envs
        self.envs = [env(env_config) for _ in range(n_envs)]
        
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
    
    def reset(self, indices=None, seeds=None):
       
        if indices is None:
            indices = range(self.n_envs)
        
        if seeds is not None:
            assert len(indices) == len(seeds), 'Seeds and indices lengths do not match.'
        
        for i, idx in enumerate(indices):
            seed = seeds[i] if seeds is not None else None
            _ = self.envs[idx].reset(seed=seed)
                    
        return self.observe()
    
    def step(self, actions):
        "Perform a step in each environment instance sequentially."
        assert len(actions) == self.n_envs, "Incorrect number of actions given. Actions should be given as np.array or List."
        
        observations, rewards, terminateds, truncateds, infos = [], [], [], [], [] 
        
        for idx, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[idx])
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)            
            
        return np.array(observations), np.array(rewards), np.array(terminateds), np.array(truncateds), infos
    
    def observe(self):
        return np.array([env.observe() for env in self.envs])
    
    def render(self, env_idx=0):
        return self.envs[env_idx].render()
