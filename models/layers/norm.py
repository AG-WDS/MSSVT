from functools import partial
import torch
from torch import nn
from torch.nn import functional as F

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            batch_size, channels, depth, height, width = x.shape   # (batch, channels, height, width)
            x = x.view(batch_size, channels, -1).transpose(1, 2)  # (batch, height*width, channels)=(batch, sequence num, feature dimension)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)  # 
            x = x.transpose(1, 2).view(batch_size, channels, depth, height, width) 
        elif x.ndim == 4:
            batch_size, channels, height, width = x.shape
            x = x.view(batch_size, channels, -1).transpose(1, 2)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.transpose(1, 2).view(batch_size, channels, height, width)      
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


norm_layer_dict = {
    "layernorm": partial(LayerNorm, eps=1e-6),
    "layernormbf16": partial(LayerNorm, eps=1e-5),
}

