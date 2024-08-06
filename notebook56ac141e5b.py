# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import time
import gym



def initialize_room(width=10, max_island_size=5, min_n_islands=1, max_n_islands=5):
    """
    Initialises the basic room shape.
    Room is represented by a np.array(shape=(width, width)), initially just with 0 for wall and 1 for floor.
    Start by marking walls around the edges of the square, then creates 'islands' of inaccessible areas.
    No dirt is created yet.
    """
    assert width > max_island_size

    room = np.zeros([width, width], dtype=np.int8)

    room[1:-1, 1:-1] = 1

    n_islands = np.random.randint(low=min_n_islands, high=max_n_islands)
    for island in range(n_islands):
        island_size = np.random.randint(low=1, high=max_island_size)
        island_x = np.random.randint(low=0-island_size + 1, high=width-1)
        island_y = np.random.randint(low=0-island_size + 1, high=width-1)

        for x_pos in range(island_x, island_x + island_size):
            for y_pos in range(island_y, island_y + island_size):
                if x_pos < 0:
                    x = 0
                elif x_pos >= width:
                    x = width - 1
                else:
                    x = x_pos
                if y_pos < 0:
                    y = 0
                elif y_pos >= width:
                    y = width - 1
                else:
                    y = y_pos

                room[x, y] = 0

    return room

def is_valid_room(room):
    """
    Takes in an initialised room of walls and floors, then returns True if every cell of the
    room is accessible, False otherwise.
    """
    target_sum = np.sum(room)
    visited = np.zeros(room.shape)

    if target_sum == 0:
        return False

    first_cell = np.argwhere(room==1)[0]

    def explore(room, current_cell, depth, max_depth=100):
        if depth > max_depth: return
        if visited[current_cell[0], current_cell[1]] == 1: return
        visited[current_cell[0], current_cell[1]] = 1

        neighbours = [[current_cell[0] - 1, current_cell[1]] if current_cell[0] > 0 else None,
                      [current_cell[0] + 1, current_cell[1]] if current_cell[0] < room.shape[0] else None,
                      [current_cell[0], current_cell[1] - 1] if current_cell[1] > 0 else None,
                      [current_cell[0], current_cell[1] + 1] if current_cell[1] < room.shape[1] else None]
        neighbours = [neighbour for neighbour in neighbours if neighbour is not None]
        neighbours = [neighbour if room[neighbour[0], neighbour[1]] == 1 else None for neighbour in neighbours]
        neighbours = [neighbour for neighbour in neighbours if neighbour is not None]

        for neighbour in neighbours:
            explore(room, neighbour, depth + 1)

    explore(room, first_cell, depth=0)

    return np.sum(visited) == target_sum

def generate_room(width=10, max_island_size=5, min_n_islands=1, max_n_islands=5, seed=None):
    """
    Will generate a new room with given parameters.
    After 1 million attempts, program will throw an error, as it's likely the user has entered
    invalid parameters than cannot build a valid room.
    """
    if seed is not None:
        np.random.seed(seed)

    attempts = 1
    room = initialize_room(width, max_island_size, min_n_islands, max_n_islands)
    while not is_valid_room(room):
        assert attempts < 1e6, "1e6 generations attempted, issue with generation parameters."
        attempts += 1
        room = initialize_room(width, max_island_size, min_n_islands, max_n_islands)

    if seed is not None:
        reset_rng()

    return room


def spawn_robot(room, pos_x=None, pos_y=None, orientation=None, seed=None):
    if pos_x is not None and pos_y is not None:
        assert room[pos_x, pos_y] in [-1, 1], "Invalid spawn position."
        if orientation is None:
            orientation = np.random.randint(low=1, high=5)
        room[pos_x, pos_y] += orientation
        return room

    if seed is not None:
        np.random.seed(seed)

    room_size_x, room_size_y = room.shape[0], room.shape[1]
    pos_x, pos_y = np.random.randint(low=0, high=room_size_x), np.random.randint(low=0, high=room_size_y)
    while room[pos_x, pos_y] not in [-1, 1]:
        pos_x, pos_y = np.random.randint(low=0, high=room_size_x), np.random.randint(low=0, high=room_size_y)

    # An orientation of 1 is facing upward, then moving clockwise so 4 is nine o'clock
    orientation = np.random.randint(low=1, high=5) + 1  # +1 as it's spawning on floor, which has a value of 1
    room[pos_x, pos_y] = orientation

    if seed is not None:
        reset_rng()

    return room

def spawn_dirt(room, fraction=1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    existing_dirt = (room < 0).astype(np.int8)
    dirt = np.random.uniform(size=room.shape) + existing_dirt
    dirt = 2 * (dirt > fraction).astype(np.int8) - 1

    if seed is not None:
        reset_rng()

    robot_pos = np.argwhere(abs(room) > 1)
    if len(robot_pos) == 0:
        return dirt * room
    else:
        dirty_room = dirt * room
        dirty_room[robot_pos[0][0], robot_pos[0][1]] = abs(dirty_room[robot_pos[0][0], robot_pos[0][1]])
        return dirty_room

def spawn_n_dirt(room, n=1, seed=None):
    clean_tile_indices = np.argwhere(room == 1)

    n_clean_tiles = len(clean_tile_indices)

    if n_clean_tiles == 0:
        return room

    if n_clean_tiles < n:
        n = n_clean_tiles

    if seed is not None:
        np.random.seed(seed)

    chosen_indices = clean_tile_indices[np.random.choice(len(clean_tile_indices), size=n, replace=False)]
    room[chosen_indices[:, 0], chosen_indices[:, 1]] = -1

    if seed is not None:
        reset_rng()

    return room

def clean_room(room):
    "Returns a cleaned version of the room."
    dirt = 2 * (room > 0).astype(np.int8) - 1
    return dirt * room


def construct_image(room):
    is_robot_in_room = len(np.argwhere(abs(room) > 1)) > 0

    # wall=0, clean=1, dirty=2, robot=3
    if is_robot_in_room:
        cmap = ListedColormap(['#2E282A', '#F3EFE0', '#A8664A','#80A1D4'])
    elif is_room_clean(room):
        cmap = ListedColormap(['#2E282A', '#F3EFE0'])
    else:
        cmap = ListedColormap(['#2E282A', '#F3EFE0', '#A8664A'])

    image = np.zeros(shape=(room.shape))
    image[room > 0] = 1
    image[room < 0] = 2
    image[abs(room) > 1] = 3
    return image, cmap

def calculate_robot_arrow(room):
    robot_position = np.argwhere(abs(room) > 1)[0]
    robo_orientation = abs(room[robot_position[0], robot_position[1]]) - 1

    orientation_map = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
    dx, dy = orientation_map[robo_orientation]
    arrow = patches.FancyArrow(
        robot_position[1],
        robot_position[0],
        dx/4,
        dy/4,
        width=0.12,
        head_width=0.4,
        head_length=0.2,
        color='#FFFFFF',
    )
    return arrow

def display_room(room):

    image, cmap = construct_image(room)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap, origin='lower')

    if len(np.argwhere(abs(room) > 1)) > 0:
        arrow = calculate_robot_arrow(room)
        ax.add_patch(arrow)

    plt.show()


def is_room_clean(room):
    return -1 not in room


def reset_rng():
    """
    Resets the NumPy random number generator with a new seed derived from the current time.
    This allows for unique seeding up to a 10th of a microsecond resolution within the 32-bit integer limit.
    """
    time_seed = np.random.seed(np.int64((time.time() * 1e7) % (2**32-1)))
    np.random.seed(time_seed)


def get_robot_pos(room):
    robot_pos = np.argwhere(abs(room) > 1)
    if len(robot_pos) == 0:
        return None
    else:
        return robot_pos[0]

def robo_move_forward(room):
    """
    Move robot forward one cell.
    If against wall, robot remains in place.
    Returns updated room and flag indicating if move was successful.
    """
    assert get_robot_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robot_pos(room)
    robo_orientation = room[robot_pos[0], robot_pos[1]] - 1

    # Robot face- up
    if robo_orientation == 1:
        move = np.array([1, 0])
    # Face- right
    elif robo_orientation == 2:
        move = np.array([0, 1])
    # Face- down
    elif robo_orientation == 3:
        move = np.array([-1, 0])
    # Face- left
    elif robo_orientation == 4:
        move = np.array([0, -1])

    new_pos = robot_pos + move

    if room[new_pos[0], new_pos[1]] == 0:
        return room, False

    room[new_pos[0], new_pos[1]] = room[robot_pos[0], robot_pos[1]]

    room[robot_pos[0], robot_pos[1]] = 1

    return room, True

def robo_move_backward(room):
    "Same as move forward, just reversed."
    assert get_robot_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robot_pos(room)
    robo_orientation = room[robot_pos[0], robot_pos[1]] - 1

    # Robot face- up
    if robo_orientation == 1:
        move = np.array([-1, 0])
    # Robot face- right
    elif robo_orientation == 2:
        move = np.array([0, -1])
    # Robot face- down
    elif robo_orientation == 3:
        move = np.array([1, 0])
    # Robot face- left
    elif robo_orientation == 4:
        move = np.array([0, 1])

    new_pos = robot_pos + move

    if room[new_pos[0], new_pos[1]] == 0:
        return room, False

    room[new_pos[0], new_pos[1]] = room[robot_pos[0], robot_pos[1]]

    room[robot_pos[0], robot_pos[1]] = 1

    return room, True

def robo_turn_right(room):
    "Robot rotates 90 degrees clockwise."
    assert get_robot_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robot_pos(room)
    robo_orientation = room[robot_pos[0], robot_pos[1]] - 2  

    new_orientation = ((robo_orientation + 1) % 4) + 2
    room[robot_pos[0], robot_pos[1]] = new_orientation
    return room, True

def robo_turn_left(room):
    "Robot rotates 90 degrees in anti-clockwise direction."
    assert get_robot_pos(room) is not None, "Move not possible."
    robot_pos = get_robot_pos(room)
    robo_orientation = room[robot_pos[0], robot_pos[1]] - 2  

    new_orientation = ((robo_orientation - 1) % 4) + 2
    room[robot_pos[0], robot_pos[1]] = new_orientation
    return room, True

def robo_wait(room):
    "Robot remains stationary."
    return room, True


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
        assert (not self.is_terminated()) and (not self.is_truncated()), "Environment is done."

        
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
        "Plays current episode video from start to now."
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
        """
        In case a config file isn't provided for the environment, these are the default settings.
        """
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
        "Flag indicating environment has run for longer than max steps."
        return len(self.history) > self.max_steps

    def is_terminated(self):
        "Flag indicating room is clean."
        return -1 not in self.room


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

def build_room_permutations(room):
    permutations =  np.stack((
        room.copy(),                  
        room.copy().T[::-1],          
        room.copy()[::-1].T[::-1].T,  
        room.copy()[::-1].T,          

        room.copy().T[::-1].T,        
        room.copy().T,                
        room.copy()[::-1],            
        room.copy()[::-1].T[::-1]     
    ))

    flat_indices = np.argmax(permutations.reshape(test_perms.shape[0], -1), axis=1)
    robot_positions = np.column_stack(np.unravel_index(flat_indices, permutations.shape[1:]))

    rotations = np.array([0, 1, 2, 3, 0, 1, 2, 3])  
    permutations[np.arange(8), robot_positions[:, 0], robot_positions[:, 1]] = (
        (permutations[np.arange(8), robot_positions[:, 0], robot_positions[:, 1]] - 2 + rotations) % 4) + 2

    return permutations

def display_room_permutations(room):
    "Displays the original room, and the 7 identical permutations of that room."
    room_list = [
        room.copy(),
        room.copy().T[::-1],
        room.copy()[::-1].T[::-1].T,
        room.copy()[::-1].T,

        room.copy().T[::-1].T,
        room.copy().T,
        room.copy()[::-1],
        room.copy()[::-1].T[::-1]
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for idx, room in enumerate(room_list):

        robot_position = np.argwhere(abs(room) > 1)[0]
        current_orientation = room[robot_position[0], robot_position[1]]
        new_orientation = (current_orientation + idx - 2) % 4 + 2
        room[robot_position[0], robot_position[1]] = new_orientation

        image, cmap = construct_image(room)

        axes.flatten()[idx].imshow(image, cmap=cmap, origin='lower')

        if len(np.argwhere(abs(room) > 1)) > 0:
            arrow = calculate_robot_arrow(room)
            axes.flatten()[idx].add_patch(arrow)

    plt.tight_layout()
    plt.show()

display_room_permutations(env.room)

env_config = {
    'width': 10,
    'max_island_size': 5,
    'min_n_islands': 1,
    'max_n_islands': 5,
    'dirt_fraction': 0.5,
    'n_dirt_generation': True,
    'n_dirt_tiles': 1,
    'seed': 42,
    'max_steps': 1000,
    'sparse_reward': True,
}

env = CleaningRobots(env_config)

env.step(2)
for _ in range(4):
    env.step(0)
env.step(2)
for _ in range(3):
    env.step(0)
env.step(2)
for _ in range(5):
    env.step(0)

animation = env.render()
HTML(animation.to_jshtml())

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


envs = EnvClusterManager(CleaningRobots, n_envs=8, env_config=env_config)

random_agent = RandomAgent()
random_agent.set_envs(envs)

mean, std = random_agent.evaluate_policy(100)
print(f"Policy mean reward: {mean}, std: {std}")

import torch
import torch.nn.functional as F
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        image_shape = (10, 10)
        kernel_size = 3
        stride = 1
        padding = 1


        expansion_ratio = 4


        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise_expand = nn.Conv2d(in_channels, out_channels*expansion_ratio, kernel_size=1, bias=False)
        self.pointwise_contract = nn.Conv2d(out_channels*expansion_ratio, out_channels, kernel_size=1, bias=False)


        if in_channels != out_channels:
            self.residual_pathway = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_pathway = nn.Identity()

        self.activation = nn.ReLU()

    def forward(self, x):

        residual = self.residual_pathway(x)


        x = self.depthwise_conv(x)

        x = self.pointwise_expand(x)
        x = self.activation(x)

        x = self.pointwise_contract(x)
        x = self.activation(x)

        return x + residual

class Network(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        n_filters  = 32 if config == None else config['n_filters']
        n_blocks   =  6 if config == None else config['n_blocks']
        n_actions  =  5 if config == None else config['n_actions']
        image_size = 10 if config == None else config['image_size']

        conv_output_shape = (n_filters, image_size, image_size)

        blocks = [ResidualBlock(in_channels=6, out_channels=n_filters)]
        blocks.extend([ResidualBlock(n_filters, n_filters) for _ in range(n_blocks - 1)])

        self.base = nn.Sequential(
            *blocks,
            nn.Flatten()
        )

        self.policy_head = nn.Linear(np.prod(conv_output_shape), n_actions)  
        self.value_head = nn.Linear(np.prod(conv_output_shape), 1)           

    def forward(self, x):
        "Forward pass through base, policy and value heads"
        x = self.base(x)
        return self.policy_head(x), self.value_head(x)

network_config = {
    'n_filters': 32,   
    'n_blocks': 6,     
    'n_actions': 5,    
    'image_size': 10,  
}

input_tensor = torch.tensor(np.ones((2, 6, 10, 10)), dtype=torch.float32)

ntwrk = Network(network_config)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shapes: {ntwrk(input_tensor)[0].shape}, {ntwrk(input_tensor)[1].shape}")

total_params = sum(p.numel() for p in ntwrk.parameters())
print(f"Network params: {total_params}")

import os
import csv

class Logger:
    def __init__(self, logs_filename='logs'):
        current_dir = os.getcwd()
        log_dir = os.path.join(current_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.logs_path = os.path.join(log_dir, logs_filename + '.csv')

    def log(self, data: dict):
        "data: dict containing name and value of each logged variable"
        file_exists = os.path.isfile(self.logs_path)
        with open(self.logs_path, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

    def delete_logs(self, verbose=True):
        if os.path.exists(self.logs_path):
            os.remove(self.logs_path)
            if verbose: print(f"Deleted log file at: {self.logs_path}")
        else:
            if verbose: print(f"No log file found at: {self.logs_path}")

import os
import time
from collections import deque

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
                    buffer_values[step]   = ntwrk.value_head(enc).view(-1)     

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

        config['batch_size'] = config['n_envs'] * config['n_rollout_steps']
        config['minibatch_size'] = config['batch_size'] // config['n_minibatches']
        config['n_epochs'] = config['training_steps'] // config['batch_size']

        return config

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

ppo_config['batch_size'] = ppo_config['n_envs'] * ppo_config['n_rollout_steps']
ppo_config['minibatch_size'] = ppo_config['batch_size'] // ppo_config['n_minibatches']
ppo_config['n_epochs'] = ppo_config['training_steps'] // ppo_config['batch_size']

print(f"Batch size: {ppo_config['batch_size']}")
print(f"Minibatch size: {ppo_config['minibatch_size']}")

def validate_ppo_config(ppo_config):
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

envs = EnvClusterManager(CleaningRobots, n_envs=1, env_config=env_config)
done = False

while not done:
    actions = agent.select_action(envs.observe(), deterministic=True)
    states, rewards, terminateds, truncateds, infos = envs.step(actions)
    done = True in np.logical_or(terminateds, truncateds)

print('done')

animation = envs.render()
HTML(animation.to_jshtml())

import numpy as np
import pandas as pd
import csv
import os
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt

refresh_duration = 60
theme = 'dark'
logs_filename = 'cleaning-robots-experiment-1-logs.csv'

current_dir = os.getcwd()
log_dir = os.path.join(current_dir, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logs_path = os.path.join(log_dir, logs_filename)
if not os.path.exists(logs_path):
    print(f"Error: The file '{logs_path}' was not found.\n")

    file_found = False
    print("Listing all folders and files in current directory:")
    for dirname, dirs, filenames in os.walk(current_dir):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            file_found = True
    if not file_found:
        print("No files were found in the working directory.")
    raise FileNotFoundError(f"The logs file '{logs_path}' was not found.")

plt.style.use('ggplot')
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
if theme == 'dark':
    plt.rcParams['figure.facecolor'] = '#171717'
    plt.rcParams['axes.facecolor'] = '#222222'
    plt.rcParams['text.color'] = '#DDDDDD'
    plt.rcParams['axes.labelcolor'] = '#999999'
    plt.rcParams['xtick.color'] = '#999999'
    plt.rcParams['ytick.color'] = '#999999'
    plt.rcParams['grid.color'] = '#444444'
    plt.rcParams['axes.edgecolor'] = '#DDDDDD'
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 9

def plot_data(df, window=5, plot_raw_points=False):
    def dynamic_wma(x):
        actual_window = len(x)
        weights = np.ones(actual_window)
        dynamic_weights = weights / sum(weights)
        return np.sum(dynamic_weights * x)

    cols = df.columns
    exclude_cols = ['timestep']
    plot_cols = [col for col in cols if col not in exclude_cols]
    plot_height = 3.5

    plt.figure(figsize=(12, plot_height * len(plot_cols)))

    for idx, col in enumerate(plot_cols):
        series = df[col]
        trend_line = series.rolling(window=window, center=True, min_periods=1).apply(dynamic_wma, raw=True)
        variance = series.rolling(window=window, center=True, min_periods=1).std()

        plt.subplot(len(plot_cols), 1, idx + 1)
        plt.fill_between(series.index, trend_line - variance, trend_line + variance, color='#666666', alpha=0.3, label=f'{col} Variance')
        plt.plot(series.index, trend_line, label=f'{col} Trend Line', color=colors[idx % len(colors)])
        if plot_raw_points:
            plt.scatter(series.index, series, alpha=0.5, s=6, label=f'{col} Original', color=colors[idx % len(colors)])

        plt.xlabel('Timestep', loc='center')
        plt.title(f'{col}', y=1.02)

    plt.tight_layout()
    plt.show()

df = pd.read_csv(logs_path)
clear_output(wait=True)
plot_data(df, plot_raw_points=False)
