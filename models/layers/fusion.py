import torch
from torch import nn
from .norm import norm_layer_dict


class BandFusion(nn.Module):
    """
    Fuses spectral features from 3D to 2D.
    
    This module reduces the spectral depth (D) to 1 using a 3D convolution,
    followed by an MLP residual block for feature refinement.
    
    Args:
        in_channels (int): Input channel dimension.
        feature_depth (int): The spectral depth size to be compressed.
        norm_layer (str): Normalization layer type.
    """
    def __init__(self, in_channels: int, feature_depth: int, norm_layer:str):
        super().__init__()

        self.spectral_projector = nn.Conv3d(
                in_channels=in_channels, out_channels=in_channels,
                kernel_size=(feature_depth, 1, 1),
                padding=0
        )
        
        self.mlp = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels, out_channels=in_channels*2,
                kernel_size=(1, 1, 1),
            ),
            norm_layer_dict[norm_layer](in_channels*2),
            nn.SiLU(),
            
            nn.Conv3d(
                in_channels=in_channels*2, out_channels=in_channels,
                kernel_size=(1, 1, 1),
            ),
            norm_layer_dict[norm_layer](in_channels),
        )
        
        self.final_SiLU = nn.SiLU()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        identity = self.spectral_projector(x)
        
        x_mlp = self.mlp(identity)
        
        x_out = identity + x_mlp
        
        x_out = self.final_SiLU(x_out) # (B, C, 1, H, W)
        
        return x_out.squeeze(2) # (B, C, H, W)


class GatedFusion(nn.Module):
    """
    Gated Fusion Module.
    
    Computes a learnable weighted sum of input features and adds it to the main feature (residual connection).
    Weights are learned via a softmax gate.
    """
    def __init__(self, num_inputs: int):
        super().__init__()
        self.gates = nn.Parameter(torch.zeros(num_inputs, 1, 1, 1))
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x: torch.Tensor, features: tuple[torch.Tensor]):
        weights = self.softmax(self.gates)
        stacked_features = torch.stack(features, dim=0)
        fused_sum = (stacked_features * weights.unsqueeze(1)).sum(dim=0)
        return x + fused_sum


class GatedFusionPassthrough(nn.Module):
    """
    Identity/Passthrough module mimicking GatedFusion API.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x, features):
        return x
    

class MasterFusion(nn.Module):
    """
    Concatenates features from multiple sources and projects them to the target dimension.
    """
    def __init__(self, embed_dim: int, input_num: int, norm_layer: str):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim * input_num, embed_dim, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer_dict[norm_layer](embed_dim),
            nn.SiLU(),
        )
    
    def forward(self, features: torch.Tensor):

        x = torch.concat(features, dim=1)
        return self.mlp(x)


class SensorFusion(nn.Module):
    """
    Hierarchical Sensor Fusion Module.
    
    Applies MasterFusion across multiple pyramid stages (typically 4 stages).
    """
    def __init__(self, embedding_list: list, input_num: int, norm_layer: str):
        super().__init__()
        self.fusion_stages = nn.ModuleList(
            [MasterFusion(embedding_list[i], input_num, norm_layer) for i in range(4)]
        )
        
    def forward(self, features: tuple[list[torch.Tensor]]) -> list[torch.Tensor]:
            """
            Args:
                features: A tuple where each element is a list of tensors for that specific stage.
                        Format: ( [Stage0_BranchA, Stage0_BranchB], [Stage1_BranchA, ...], ... )
            """
            stages_to_fuse = zip(*features)
            
            fused_features = []
            
            for stage_data, fusion_module in zip(stages_to_fuse, self.fusion_stages):
                fused_stage = fusion_module(stage_data)
                fused_features.append(fused_stage)
                
            return fused_features
    