import torch
from torch import nn
from .norm import norm_layer_dict
import math


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer:str, 
                 kernel_size: tuple = (1, 3, 3),  # (7, 4, 3) (3, 2, 1)
                 stride: tuple = (1, 2, 2), 
                 padding: tuple = (0, 1, 1),):
        super().__init__()
        
        self.patch_embedding = nn.Conv3d(in_channels, out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm_layer_dict[norm_layer](out_channels)

    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        x = self.norm(x)
        return x
    

class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer:str, 
                 kernel_size: tuple = (1, 3, 3),  # (7, 4, 3) (3, 2, 1)
                 stride: tuple = (1, 2, 2), 
                 padding: tuple = (0, 1, 1),):
        super().__init__()
        if len(kernel_size) == 3:
            kernel_size = kernel_size[1:]
            stride = stride[1:]
            padding = padding[1:]
        self.patch_embedding = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm_layer_dict[norm_layer](out_channels)

    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        x = self.norm(x)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer:str, 
                 kernel_size: tuple = (1, 3, 3),  # (7, 4, 3) (3, 2, 1)
                 stride: tuple = (1, 2, 2), 
                 padding: tuple = (0, 1, 1),
                 ndim="3D"):
        super().__init__()
        self.ndim = ndim
        if self.ndim == "3D":
            self.patch_embedding = PatchEmbed3D(in_channels, out_channels, norm_layer, kernel_size, stride, padding)
        else:
            self.patch_embedding = PatchEmbed2D(in_channels, out_channels, norm_layer, kernel_size, stride, padding)
        

    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        return x

