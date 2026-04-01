import torch
from torch import nn
import math

class SegmentationHead(nn.Module):
    def __init__(self, embed_dim: int, scale_factor: float, num_classes: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.apply(self._init_weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(self.conv(x))

    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()