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
