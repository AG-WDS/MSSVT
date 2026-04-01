import torch
from torch import nn
from .norm import norm_layer_dict


class MLP2D(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, norm_layer: str, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=1)

        self.norm = norm_layer_dict[norm_layer](embed_dim)

    def forward(self, x):

        x = self.proj(x)
        x = self.norm(x)

        return x

