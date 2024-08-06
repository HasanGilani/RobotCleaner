import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import time
import gym

def initialize_room(width=10, max_island_size=5, min_n_islands=1, max_n_islands=5):
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

    orientation = np.random.randint(low=1, high=5) + 1  
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
    robot_orientation = abs(room[robot_position[0], robot_position[1]]) - 1

    orientation_map = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
    dx, dy = orientation_map[robot_orientation]
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
    time_seed = np.random.seed(np.int64((time.time() * 1e7) % (2**32-1)))
    np.random.seed(time_seed)
    
    
    
def get_robo_pos(room):
    robot_pos = np.argwhere(abs(room) > 1)
    if len(robot_pos) == 0:
        return None
    else:
        return robot_pos[0]
    
def robo_move_forward(room):
    assert get_robo_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robo_pos(room)
    robot_orientation = room[robot_pos[0], robot_pos[1]] - 1
    
    if robot_orientation == 1: 
        move = np.array([1, 0])
    elif robot_orientation == 2:
        move = np.array([0, 1])
    elif robot_orientation == 3:
        move = np.array([-1, 0])
    elif robot_orientation == 4:
        move = np.array([0, -1])
        
    new_pos = robot_pos + move
    
    if room[new_pos[0], new_pos[1]] == 0:
        return room, False
    
    room[new_pos[0], new_pos[1]] = room[robot_pos[0], robot_pos[1]]
    
    room[robot_pos[0], robot_pos[1]] = 1
    
    return room, True

def robo_move_backward(room):
    "Same as move forward, just reversed."
    assert get_robo_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robo_pos(room)
    robot_orientation = room[robot_pos[0], robot_pos[1]] - 1
    

    if robot_orientation == 1: 
        move = np.array([-1, 0])
    elif robot_orientation == 2:
        move = np.array([0, -1])
    elif robot_orientation == 3:
        move = np.array([1, 0])
    elif robot_orientation == 4:
        move = np.array([0, 1])
        
    new_pos = robot_pos + move
    
    if room[new_pos[0], new_pos[1]] == 0:
        return room, False
    
    room[new_pos[0], new_pos[1]] = room[robot_pos[0], robot_pos[1]]
    
    room[robot_pos[0], robot_pos[1]] = 1
    
    return room, True

def robo_turn_right(room):
    "Robot rotates 90 degrees clockwise."
    assert get_robo_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robo_pos(room)
    robot_orientation = room[robot_pos[0], robot_pos[1]] - 2  
    
    new_orientation = ((robot_orientation + 1) % 4) + 2
    room[robot_pos[0], robot_pos[1]] = new_orientation
    return room, True

def robo_turn_left(room):
    "Robot rotates 90 degrees anti-clockwise."
    assert get_robo_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robo_pos(room)
    robot_orientation = room[robot_pos[0], robot_pos[1]] - 2  
    
    new_orientation = ((robot_orientation - 1) % 4) + 2
    room[robot_pos[0], robot_pos[1]] = new_orientation
    return room, True

def robo_wait(room):
    "Robot doesn't perform any actions and just remains where it is."
    return room, True


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
