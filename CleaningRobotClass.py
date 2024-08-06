import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import time
import gym
from robot_functions_module import (
    initialize_room,
    is_valid_room,
    generate_room,
    spawn_robot,
    spawn_dirt,
    spawn_n_dirt,
    clean_room,
    construct_image,
    calculate_robot_arrow,
    display_room,
    is_room_clean,
    reset_rng,
    robo_move_forward,
    robo_move_backward,
    robo_turn_right,
    robo_turn_left,
    robo_wait,
    get_robo_pos
)

class CleaningRobots:
    def __init__(self, config=None):
        if config == None:
            self.config = self.default_config()
        else:
            self.config = config
        
        self.max_steps = self.config['max_steps']
        self.sparse_reward = self.config['sparse_reward']
        
        self.room = self.initialize_environment()
        
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=-1, high=5, shape=(6, self.config['width'], self.config['width']), dtype=np.int8)

        self.history = []
        self.history.append(self.room.copy())

    
    def observe(self, decompose_channels=True):
        if decompose_channels:
            "Splits room up into several channels for an agent to learn from."
            acc_chanl = np.where((self.room != 0), 1, 0)                  
            dirt_chanl = np.where((self.room == -1), 1, 0)                       

            robo_orientation = np.max(self.room) - 2                              
            robo_pos = np.argwhere(self.room == np.max(self.room))[0]             
            robo_pos_chanl = np.zeros((4, *self.room.shape))                   
            robo_pos_chanl[robo_orientation, robo_pos[0], robo_pos[1]] = 1  

            return np.concatenate((np.stack((acc_chanl, dirt_chanl)), robo_pos_chanl), axis=0)
        else:
            return self.room.copy()
        
    def reset(self, seed=None):
        "Resets environment, generating a new room to clean according to config."
        self.config['seed'] = seed
        self.room = self.initialize_environment()
        self.history = []
        self.history.append(self.room.copy())
        return self.observe()
    
    def step(self, action):
        assert (not self.is_terminated()) and (not self.is_truncated()), "Environment done."
        
        if action == 0:
            self.room, action_success = robo_move_forward(self.room)
        
        elif action == 1:
            self.room, action_success = robo_move_backward(self.room)
        
        elif action == 2:
            self.room, action_success = robo_turn_right(self.room)
        
        elif action == 3:
            self.room, action_success = robo_turn_left(self.room)
        
        elif action == 4:
            self.room, action_success = robo_wait(self.room)
        
        self.history.append(self.room.copy()) 
        
        reward = self.calculate_reward(action, action_success)
        
        info = {'step': len(self.history)}
        
        return self.observe(), reward, self.is_terminated(), self.is_truncated() , info
            
    def render(self):
        "Plays video of current episode from start to now."
        fig, ax = plt.subplots()

        image_data, cmap = construct_image(self.history[0])
        img = ax.imshow(image_data, cmap=cmap, origin='lower')

        arrow_patch = None

        def update(frame):
            nonlocal arrow_patch
            image_data, cmap = construct_image(self.history[frame])
            arrow_data = calculate_robot_arrow(self.history[frame])
            img.set_data(image_data)

            if arrow_patch is not None:
                arrow_patch.remove()

            arrow_patch = ax.add_patch(arrow_data)

            return [img, arrow_patch]

        ani = FuncAnimation(fig, update, frames=range(len(self.history)), interval=400, blit=True)
        plt.close()
        return ani
    
    def calculate_reward(self, action, action_success=True):
        assert self.config['sparse_reward'], 'Dense reward not implemented yet.'
        
        if self.config['sparse_reward']:
            max_room_size = self.room.reshape(-1).shape[0]
            action_penalty_scale = 1 / (1 * (max_room_size * 2))
            
            if action in [0, 2, 3, 4]:
                action_penalty = -1 * action_penalty_scale
            elif action in [1]:
                action_penalty = -1.25 * action_penalty_scale
                
            if action_success == False:
                action_penalty -= 0.1
                
            if self.is_terminated():
                reward = 1.
                
            else:
                reward = 0.
                
            return reward + action_penalty

    #Initializing environment for the robot to clean the area
    def initialize_environment(self, attempts=0):
        self.room = generate_room(width=self.config['width'], 
                                  max_island_size=self.config['max_island_size'], 
                                  min_n_islands=self.config['min_n_islands'], 
                                  max_n_islands=self.config['max_n_islands'],
                                  seed=self.config['seed'])
        
        self.room = spawn_robot(self.room, 
                                seed=self.config['seed'])
        
        if self.config['n_dirt_generation']:
            self.room = spawn_n_dirt(self.room, 
                                     n=self.config['n_dirt_tiles'],
                                     seed=self.config['seed'])
        else:
            self.room = spawn_dirt(self.room, 
                               fraction=self.config['dirt_fraction'],
                               seed=self.config['seed'])
            
        if attempts > 1e4:
            raise Exception("Max number of attempts to initialise environment exceeded.")
            
        if self.is_terminated():
            self.initialize_environment(attempts + 1)
            
        return self.room

    def default_config(self):
        config = {
            'width': 10,
            'max_island_size': 5,
            'min_n_islands': 1,
            'max_n_islands': 5,
            'dirt_fraction': 0.5,
            'n_dirt_generation': False,
            'n_dirt_tiles': 5,
            'seed': None,
            'max_steps': 1000,
            'sparse_reward': True,
        }
        return config
    
    def is_truncated(self):
        "If run for longer than max steps."
        return len(self.history) > self.max_steps
    
    def is_terminated(self):
        "If room is clean."
        return -1 not in self.room
