import torch
from torch import nn
import math
from .mlp import MLP2D
import torch.nn.functional as F


class SegFormerDecoder(nn.Module):
    """
    SegFormer Decoder Implementation.
    Paper: SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(
        self,
        input_feature_dims: list = [32, 64, 128, 256],
        decoder_embedding_dim: int = 256,
        dropout: float = 0.0,
        norm_layer: str = "layernormbf16",
        stage_num: int = 4,
    ):
        super().__init__()
        self.mlp_list = nn.ModuleList()
        for i in range(stage_num):
            self.mlp_list.append(
                MLP2D(norm_layer=norm_layer, input_dim=input_feature_dims[i], embed_dim=decoder_embedding_dim)
            )

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=stage_num * decoder_embedding_dim, out_channels=decoder_embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_embedding_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: list[torch.Tensor]):
        c1_size = x[0].size()[-2:]
        for i in range(len(x)):
            x[i] = self.mlp_list[i](x[i])
            x[i] = F.interpolate(x[i], size=c1_size, mode="bilinear", align_corners=False)
        _c = self.linear_fuse(torch.cat(x, dim=1))
        x = self.dropout(_c)
        return x
    
    