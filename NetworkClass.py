import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from ResidualBlockClass import ResidualBlock

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
        x = self.base(x)
        return self.policy_head(x), self.value_head(x)
