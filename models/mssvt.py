from copy import deepcopy
import torch
import torch.nn as nn

from .layers.blocks import SpeMambaBlock, SpaBandBlock
from .layers.patch_embed import PatchEmbed
from .layers.band_grouping import BandMulGroupSplitter2D3D
from .layers.fusion import BandFusion, SensorFusion, GatedFusion, GatedFusionPassthrough
from .layers.head import SegmentationHead
from .layers.segformer_decoder import SegFormerDecoder


BASE_CONFIG = {
    "in_channels": 1,
    "initial_spectral_dim": 55,
    "embed_dims": [32, 64, 160, 256],
    "spectral_dims": [55, 14, 7, 4, 2],
    "num_heads_list": [2, 4, 10, 16],
    "n_win": 8,
    "topk_list": [1, 4, 16, -2],
    "side_dwconv": 5,
    "kv_downsample_ratios": [8, 4, 2, 1],
    "kv_per_wins": [2, 2, 2, -1],
    "patch_spatial_kernel_size_list": [(7,7,7), (3,3,3), (3,3,3), (3,3,3)],
    "patch_spatial_stride_list": [(4,4,4), (2,2,2), (2,2,2), (2,2,2)],
    "patch_spatial_padding_list": [(3,3,3), (1,1,1), (1,1,1), (1,1,1)],
    "num_blocks": [2, 2, 2, 2],
    "norm_layer": "layernormbf16",
    "align_to": 64,
    "qkv_bias": True,
    "proj_bias": True,
    "ffn_bias": True,
    "ffn_ratio": 2.0,
    "drop_path_rate": 0.30,
    "dropout": 0.15,
}

FEATURE_CONFIGS = {
    "Sensor-RGB": {
        "in_channels": 1,
        "initial_spectral_dim": 38,
        "spectral_dims": [38, 10, 5, 3, 2],
        "patch_spatial_kernel_size_list": [(7,7,7), (3,3,3), (3,3,3), (3,3,3)],
        "patch_spatial_stride_list": [(4,4,4), (2,2,2), (2,2,2), (2,2,2)],
        "patch_spatial_padding_list": [(3,3,3), (1,1,1), (1,1,1), (1,1,1)],
        "ndim": "3D",
        "mode": "3DSpaBandBlock",
    },
    "color": {
        "in_channels": 10,
        "patch_spatial_kernel_size_list": [(7,7,7), (3,3,3), (3,3,3), (1,3,3)],
        "patch_spatial_stride_list": [(4,4,4), (2,2,2), (2,2,2), (1,2,2)],
        "patch_spatial_padding_list": [(3,3,3), (1,1,1), (1,1,1), (0,1,1)],
        "ndim": "2D",
        "mode": "2DSpeMamba",
    },
    "texture": {
        "in_channels": 24,
        "patch_spatial_kernel_size_list": [(7,7,7), (3,3,3), (3,3,3), (1,3,3)],
        "patch_spatial_stride_list": [(4,4,4), (2,2,2), (2,2,2), (1,2,2)],
        "patch_spatial_padding_list": [(3,3,3), (1,1,1), (1,1,1), (0,1,1)],
        "ndim": "2D",
        "mode": "2DSpeMamba",
    },
    "structure": {
        "in_channels": 1,
        "patch_spatial_kernel_size_list": [(7,7,7), (3,3,3), (3,3,3), (1,3,3)],
        "patch_spatial_stride_list": [(4,4,4), (2,2,2), (2,2,2), (1,2,2)],
        "patch_spatial_padding_list": [(3,3,3), (1,1,1), (1,1,1), (0,1,1)],
        "ndim": "2D",
        "mode": "2DSpeMamba",
    },
    "Sensor-MS": {
        "in_channels": 1,
        "initial_spectral_dim": 17,
        "spectral_dims": [17, 5, 3, 2, 1],
        "patch_spatial_kernel_size_list": [(7,7,7), (3,3,3), (3,3,3), (3,3,3)],
        "patch_spatial_stride_list": [(4,4,4), (2,2,2), (2,2,2), (2,2,2)],
        "patch_spatial_padding_list": [(3,3,3), (1,1,1), (1,1,1), (1,1,1)],
        "ndim": "3D",
        "mode": "3DSpaBandBlock",
    },
    "vis": {
        "in_channels": 13,
        "patch_spatial_kernel_size_list": [(7,7,7), (3,3,3), (3,3,3), (1,3,3)],
        "patch_spatial_stride_list": [(4,4,4), (2,2,2), (2,2,2), (1,2,2)],
        "patch_spatial_padding_list": [(3,3,3), (1,1,1), (1,1,1), (0,1,1)],
        "ndim": "2D",
        "mode": "2DSpeMamba",
    },
    "ALL": {
        "in_channels": 1,
        "initial_spectral_dim": 55,
        "spectral_dims": [55, 14, 7, 4, 2],
        "patch_spatial_kernel_size_list": [(7,7,7), (3,3,3), (3,3,3), (3,3,3)],
        "patch_spatial_stride_list": [(4,4,4), (2,2,2), (2,2,2), (2,2,2)],
        "patch_spatial_padding_list": [(3,3,3), (1,1,1), (1,1,1), (1,1,1)],
        "ndim": "3D",
        "mode": "3DSpaBandBlock",
    },
}


class Stage(nn.Module):
    """
    Single Stage Implementation containing Downsampling and Feature Processing blocks.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 in_spectral_dim: int, out_spectral_dim: int,
                 num_heads: int, 
                 n_win: int,      
                 topk: int,      
                 side_dwconv: int,
                 kv_downsample_ratio,
                 kv_per_win,
                 patch_spatial_kernel_size: tuple,
                 patch_spatial_stride: tuple,
                 patch_spatial_padding: tuple,
                 num_blocks: int,
                 norm_layer: str, align_to, qkv_bias=False, proj_bias=True, ffn_bias=True, ffn_ratio: float = 4.0, 
                 drop_path_rates_for_stage: list = None, 
                 dropout: float = 0.,
                 ndim: str = "3D",
                 mode: str = None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_spectral_dim = in_spectral_dim
        self.out_spectral_dim = out_spectral_dim

        self.dpr_idx = 0
        self.drop_path_rates_for_stage = drop_path_rates_for_stage if drop_path_rates_for_stage is not None else []

        # 1. Spatial/Spectral Downsampling (PatchEmbed3D)
        self.spatial_downsample_embed = PatchEmbed(
            in_channels, out_channels, 
            norm_layer=norm_layer,
            kernel_size=patch_spatial_kernel_size,
            stride=patch_spatial_stride,
            padding=patch_spatial_padding,
            ndim=ndim
        )
        
        # 2. Feature Modeling Blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            dpr = self.drop_path_rates_for_stage[self.dpr_idx] if self.dpr_idx < len(self.drop_path_rates_for_stage) else 0.
            if mode == "2DSpeMamba":
                self.blocks.append(SpeMambaBlock(channels=out_channels))
            elif mode == "3DSpaBandBlock":
                self.blocks.append(
                    SpaBandBlock(
                        channel_dim=out_channels, num_heads=num_heads, n_win=n_win,
                        topk=topk, side_dwconv=side_dwconv,
                        kv_downsample_ratio=kv_downsample_ratio,
                        kv_per_win=kv_per_win,
                        num_bands=out_spectral_dim, norm_layer=norm_layer, align_to=align_to,
                        qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias, ffn_ratio=ffn_ratio, drop_path=dpr,
                        dropout=dropout
                    )
                )
            else:
                raise NotImplementedError(f"Mode {mode} not implemented.")

            self.dpr_idx += 1 

    def forward(self, x: torch.Tensor):
        x = self.spatial_downsample_embed(x)
        for block in self.blocks:
            x = block(x)
        return x


class HierarchicalSpatialBandBackbone(nn.Module):
    """
    Hierarchical Spectral-Spatial Backbone (Encoder).
    Composed of multiple Stages.
    """
    def __init__(self, in_channels: int = 1, 
                 initial_spectral_dim: int = 55,
                 embed_dims: list = [32, 64, 128, 256], 
                 spectral_dims: list = [55, 14, 7, 4, 2],
                 num_heads_list: list = [2, 4, 8, 16],
                 n_win=7,
                 topk_list=[1, 4, 16, -2],
                 side_dwconv=5,
                 kv_downsample_ratios=[4, 2, 1, 1],
                 kv_per_wins=[2, 2, 2, -1],
                 patch_spatial_kernel_size_list: list = None,
                 patch_spatial_stride_list: list = None,
                 patch_spatial_padding_list: list = None,
                 num_blocks: list = [2, 2, 2, 2],
                 norm_layer: str = "layernormbf16", 
                 align_to = 64,
                 qkv_bias=False, 
                 proj_bias=True, 
                 ffn_bias=True, 
                 ffn_ratio: float = 4.0, 
                 drop_path_rate: float = 0.5,
                 dropout: float = 0.,
                 ndim: str = "3D",
                 mode: str = None,
        ): 
        super().__init__()
        
        # Validation
        assert len(embed_dims) == len(spectral_dims) - 1
        assert len(patch_spatial_kernel_size_list) == len(embed_dims)

        # DropPath configuration
        total_blocks_with_droppath = sum(num_blocks)
        dpr_rates_global = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks_with_droppath)]
        dpr_idx_counter = 0

        self.stages = nn.ModuleList()
        current_in_channels = in_channels
        current_spectral_dim = initial_spectral_dim
        
        for i in range(len(embed_dims)):
            stage_out_channels = embed_dims[i]
            stage_out_spectral_dim = spectral_dims[i+1]

            # Assign DropPath rates to this stage
            num_blocks_in_this_stage = num_blocks[i]
            dpr_rates_for_this_stage = dpr_rates_global[dpr_idx_counter : dpr_idx_counter + num_blocks_in_this_stage]
            dpr_idx_counter += num_blocks_in_this_stage
            
            self.stages.append(
                Stage(
                    in_channels=current_in_channels,
                    out_channels=stage_out_channels,
                    in_spectral_dim=current_spectral_dim,
                    out_spectral_dim=stage_out_spectral_dim,
                    num_heads=num_heads_list[i],
                    n_win=n_win,
                    topk=topk_list[i],
                    side_dwconv=side_dwconv,
                    kv_downsample_ratio=kv_downsample_ratios[i],
                    kv_per_win=kv_per_wins[i],
                    patch_spatial_kernel_size=patch_spatial_kernel_size_list[i],
                    patch_spatial_stride=patch_spatial_stride_list[i],
                    patch_spatial_padding=patch_spatial_padding_list[i],
                    num_blocks=num_blocks[i],
                    norm_layer=norm_layer,
                    align_to=align_to,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    ffn_ratio=ffn_ratio,
                    drop_path_rates_for_stage=dpr_rates_for_this_stage,
                    dropout=dropout,
                    ndim=ndim,
                    mode=mode,
                )
            )
            current_in_channels = stage_out_channels 
            current_spectral_dim = stage_out_spectral_dim 

    def forward_slave(self, x: torch.Tensor, fusion_models: list[nn.Module]) -> list[torch.Tensor]:
        out = []
        for stage, fusion in zip(self.stages, fusion_models):
            x = stage(x)
            fusion_x = fusion(x)
            out.append(fusion_x)
        return out

    def forward_master(self, 
                       x: torch.Tensor, 
                       aux_heads: list[nn.Module],
                       band_fusion_models: list[nn.Module],
                       gated_fusion_models: list[nn.Module],
                       features: tuple[tuple[torch.Tensor]]) -> list[torch.Tensor]:
        out = []
        aux_out = []
        modules_to_zip = (self.stages, aux_heads, band_fusion_models, gated_fusion_models)
        
        for i, (stage, aux_head, band_fusion, gated_fusion) in enumerate(zip(*modules_to_zip)):
            x = stage(x)
            x_2d = band_fusion(x)
            x_fused_2d = gated_fusion(x_2d, list(features[i]))
            
            if not isinstance(aux_head, nn.Identity):
                aux_out.append(aux_head(x_fused_2d))
            out.append(x_fused_2d)

        return out, aux_out

    def forward(self, x: torch.Tensor, *args) -> list[torch.Tensor]:
        if len(args) == 1:
            band_fusion = args[0]
            return self.forward_slave(x, band_fusion)
        elif len(args) == 4:
            aux_heads = args[0]
            band_fusion_models = args[1]
            gated_fusion_models = args[2]
            features = args[3]
            return self.forward_master(x, aux_heads, band_fusion_models, gated_fusion_models, features)
        else:
            raise ValueError(f"HierarchicalSpatialBandBackbone.forward received invalid args length: {len(args)}")    


def build_backbone_from_configs(feature_name: str, base_config: dict, feature_config: dict):
    if feature_name not in feature_config:
        raise ValueError(f"Feature '{feature_name}' not found in the config.")
    cfg = deepcopy(base_config)
    cfg.update(feature_config[feature_name])
    return HierarchicalSpatialBandBackbone(**cfg)


class MSSVT(nn.Module):
    """
    MSSVT: Main Model Architecture.
    """
    def __init__(self,
                 split_scheme: dict = {
                    "Sensor-RGB": (((0, 13), (30, 55)), '3D'), # 38
                    "color": (((3, 13), ), '2D'), # 10
                    "texture": (((30, 54), ), '2D'), # 24
                    "structure": (((54, 55), ), '2D'), # 1
                    "Sensor-MS": (((13, 30), ), '3D'), # 17
                    "vis": (((17, 30), ), '2D'), # 13
                    "ALL": (((0, 55), ), "3D"),
                    },
                 branches: dict = {
                    'Sensor-RGB': ("color", "texture", "structure"),
                    "Sensor-MS": ("vis", ),
                    "ALL": tuple(), 
                    },
                 decoder_embedding_dim: int = 256,
                 num_classes: int = 7,
                 stage_to_supervise = [2, 3],
                 ):
        super().__init__()
        
        self.spliter = BandMulGroupSplitter2D3D(split_scheme)
        self.feature_names = split_scheme.keys()
        self.branches = branches
        self.stage_to_supervise = stage_to_supervise

        # Build Branches
        self.slave_branches = nn.ModuleDict()
        self.main_branches = nn.ModuleDict()
        
        for main_name, values in self.branches.items():
            self.main_branches[main_name] = build_backbone_from_configs(main_name, BASE_CONFIG, FEATURE_CONFIGS)
            for name in values:
                self.slave_branches[name] = build_backbone_from_configs(name, BASE_CONFIG, FEATURE_CONFIGS)

        # Band Fusion Layers
        self.band_fusion = nn.ModuleDict()
        for name, values in split_scheme.items():
            if values[1] == '3D':
                self.band_fusion[name] = nn.ModuleList(
                    [BandFusion(BASE_CONFIG['embed_dims'][i], FEATURE_CONFIGS[name]['spectral_dims'][i+1], 
                            norm_layer=BASE_CONFIG['norm_layer']) 
                    for i in range(4)]
                )
            else:
                self.band_fusion[name] = nn.ModuleList([nn.Identity() for _ in range(4)])
        
        # Master and Sub Branch Fusion
        self.gated_fusion = nn.ModuleDict()
        for main_name, values in self.branches.items():
            self.gated_fusion[main_name] = nn.ModuleList(
                [GatedFusion(len(values)) if len(values) else GatedFusionPassthrough() for _ in range(4)]
            )

        self.sensor_fusion = SensorFusion(BASE_CONFIG['embed_dims'], len(self.branches), BASE_CONFIG['norm_layer'])

        # Aux Heads
        self.aux_head = nn.ModuleDict()
        for name in self.branches.keys():
            self.aux_head[name] = nn.ModuleList(
                [SegmentationHead(embed_dim=BASE_CONFIG['embed_dims'][i],
                                  scale_factor=2 ** (i + 2),
                                  num_classes=num_classes)
                if i in self.stage_to_supervise else nn.Identity()
                                  for i in range(4)]
            )

        self.decoder = SegFormerDecoder(
            input_feature_dims=BASE_CONFIG['embed_dims'],
            decoder_embedding_dim=decoder_embedding_dim,
            dropout=BASE_CONFIG["dropout"],
            norm_layer=BASE_CONFIG["norm_layer"],
        )

        self.segmentation_head = SegmentationHead(
            embed_dim=decoder_embedding_dim,
            num_classes=num_classes,
            scale_factor=4.0,
        )

    def forward(self, x) -> torch.Tensor:
        x = self.spliter(x)
        main_outputs = []
        aux_outputs: dict[str: list[torch.Tensor]] = dict()
        
        for main_name, slave_names in self.branches.items():
            # Forward pass for slave branches
            for slave_name in slave_names:
                x[slave_name] = self.slave_branches[slave_name](x[slave_name], self.band_fusion[slave_name])

            # Gather features
            features = tuple(tuple(x[name][stage_id] for name in slave_names) for stage_id in range(4))
            
            # Forward pass for main branch
            x[main_name], aux_outputs[main_name] = self.main_branches[main_name](
                x[main_name], 
                self.aux_head[main_name], 
                self.band_fusion[main_name],
                self.gated_fusion[main_name], 
                features
            )
            main_outputs.append(x[main_name])
        
        # Final Fusion and Decoding
        x = self.sensor_fusion(tuple(main_outputs))
        x = self.decoder(x)
        x = self.segmentation_head(x)

        return x, aux_outputs
    
