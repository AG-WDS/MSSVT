import math
import torch
from torch import nn
from torch.nn import functional as F
from mamba_ssm import Mamba
from einops import rearrange
from .ss2d import SS2D
from .drop import DropPath
from .norm import norm_layer_dict
from .ffn_layers import SwiGLUFFN
from .ops.bra_legacy import BiLevelRoutingAttention


class SpaBandBlock(nn.Module):
    def __init__(self, 
                 channel_dim: int, 
                 num_heads: int, 
                 num_bands: int,
                 **kwargs):
        """
        Bi-Spectral Spatial Attention Block V2.
        
        This block integrates both spatial and spectral attention mechanisms.
        
        Args:
            channel_dim (int): Input channel dimension.
            num_heads (int): Number of attention heads.
            num_bands (int): Number of spectral bands (used for 3D data).
            **kwargs: Additional arguments passed to the sub-blocks (BiFormer parameters),
                      such as 'n_win', 'topk', 'side_dwconv', 'norm_layer', etc.
                      This design reduces code redundancy.
        """
        super().__init__()
        
        # Spatial Attention Block
        self.spatial_block = DWConv_SpatialBlock(
            channel_dim=channel_dim, 
            num_heads=num_heads, 
            **kwargs 
        )
        
        self.use_band_block = num_bands > 1
        
        if self.use_band_block:
            self.band_block = BandBlock(
                channel_dim=channel_dim, 
                num_heads=num_heads, 
                num_bands=num_bands,
                **kwargs
            )
        else:
            self.band_block = None

    def forward(self, x, pos_enc=None, **kwargs):
        x = self.spatial_block(x, pos_enc)
        if self.use_band_block:
            x = self.band_block(x, pos_enc)
        return x


class SpeMambaBlock(nn.Module):
    def __init__(self, 
                 channels: int, 
                 token_num: int = 8, 
                 use_residual: bool = True, 
                 group_num: int = 4,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2):
        """
        Spectral Mamba Block.
        
        Applies State Space Models (Mamba) along the spectral dimension.

        Args:
            channels (int): Input channel dimension.
            token_num (int): Number of tokens to split the spectral dimension into.
            use_residual (bool): Whether to add a residual connection.
            group_num (int): Number of groups for GroupNorm.
            d_state (int): SSM state expansion factor (Mamba parameter).
            d_conv (int): Local convolution width (Mamba parameter).
            expand (int): Block expansion factor (Mamba parameter).
        """
        super().__init__()
        self.channels = channels
        self.token_num = token_num
        self.use_residual = use_residual

        # Calculate grouped channel number, ensuring it can be divided evenly
        self.group_channel_num = math.ceil(channels / token_num)
        self.total_channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(
            d_model=self.group_channel_num, 
            d_state=d_state,  
            d_conv=d_conv, 
            expand=expand, 
        )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.total_channel_num),
            nn.SiLU()
        )

    def forward(self, x, *args):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # --- 1. Padding ---
        # Pad the channel dimension if the current channels are less than the required total channels.
        if C < self.total_channel_num:
            pad_c = self.total_channel_num - C
            # Create padding tensor and concatenate along the channel dimension (dim=1)
            pad_features = torch.zeros((B, pad_c, H, W), device=x.device, dtype=x.dtype)
            x_pad = torch.cat([x, pad_features], dim=1)
        else:
            x_pad = x

        # --- 2. Mamba Process (Reshape using einops) ---
        # Concept: Flatten spatial dimensions (H, W) into the batch dimension.
        #          Split the channel dimension into 'token_num' tokens.
        # Transform: (B, C, H, W) -> (B * H * W, token_num, group_channel_num)
        # where C = token_num * group_channel_num
        x_mamba_in = rearrange(x_pad, 'b (t g) h w -> (b h w) t g', t=self.token_num)
        
        # Apply Mamba scan
        x_mamba_out = self.mamba(x_mamba_in)
        
        # Restore original shape
        # Transform: (B * H * W, token_num, group_channel_num) -> (B, C, H, W)
        x_recon = rearrange(x_mamba_out, '(b h w) t g -> b (t g) h w', b=B, h=H, w=W)

        # --- 3. Projection & Residual ---
        x_proj = self.proj(x_recon)
        
        # Slice back to the original channel size if padding was applied
        if C < self.total_channel_num:
            x_proj = x_proj[:, :C, :, :]

        if self.use_residual:
            return x + x_proj
        else:
            return x_proj
        

class AttentionLePE(nn.Module):
    """
    Vanilla Attention with Locally-enhanced Positional Encoding (LePE).
    Used as a fallback or alternative to Bi-Level Routing Attention.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)

    def forward(self, x):
        """
        args:
            x: NHWC tensor
        return:
            NHWC tensor
        """
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')
        
        B, N, C = x.shape        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        lepe = self.lepe(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
        lepe = rearrange(lepe, 'n c h w -> n (h w) c')

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe

        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x
    

class VHA(nn.Module):
    """
    Core Attention Layer combining Bi-Level Routing Attention (Spatial) and SS2D (Spectral/Global).
    """
    def __init__(self, embed_dim, num_heads,
                 n_win=7,                # Region partition size (e.g., 7x7)
                 topk=4,                 # Routing sparsity (Top-k regions)
                 side_dwconv=5,          # LEPE kernel size
                 kv_downsample_ratio=4,  # KV compression ratio for memory optimization
                 kv_per_win=2,           # KV per window (used with ada_avgpool)
                 qkv_bias=True, proj_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj_bias = proj_bias
        self.qkv_bias = qkv_bias
        ds_mode = 'ada_avgpool' if kv_per_win > 0 else 'identity'

        if topk > 0:
            self.bra = BiLevelRoutingAttention(
                dim=embed_dim,
                num_heads=num_heads,
                n_win=n_win,
                topk=topk,
                side_dwconv=side_dwconv,
                qk_dim=None,                 # Default: head_dim
                qk_scale=None,               # Default: head_dim^-0.5
                param_attention="qkvo",      # Use standard QKV Linear
                param_routing=False, 
                diff_routing=False,
                soft_routing=False,
                kv_downsample_mode=ds_mode,
                kv_downsample_ratio=kv_downsample_ratio, 
                kv_per_win=kv_per_win,  
                kv_downsample_kernel=None,
                auto_pad=True
            )
        elif topk == -2:
            self.bra = AttentionLePE(dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                side_dwconv=side_dwconv)

        self.ss2d = SS2D(d_model=embed_dim)


    def forward(self, x, pos_enc=None):

        x_bra = x.permute(0, 2, 3, 1)
        x_bra = self.bra(x_bra)
        x_bra = x_bra.permute(0, 3, 1, 2)

        x_ssm = self.ss2d(x_bra)

        return x_ssm


class BandBlock(nn.Module):
    """
    Band Block.
    
    Args:
        x: Input tensor of shape (B, C, D, H, W)
    """
    def __init__(self, channel_dim: int, num_heads: int,       
                 n_win,                # Region partition size
                 topk,                 # Routing sparsity
                 side_dwconv,          # LEPE kernel size
                 kv_downsample_ratio,  # KV compression ratio
                 kv_per_win,           # KV per window
                 num_bands: int, 
                 norm_layer: str, align_to: int = 64, 
                 qkv_bias=False, proj_bias=True, ffn_bias=True, 
                 ffn_ratio: float = 4.0, drop_path: float = 0.0, dropout: float = 0.0):
        super().__init__()
        
        # Attention
        dim = channel_dim * num_bands
        self.in_channels = channel_dim*num_bands
        self.out_channels = channel_dim

        self.reduction = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.inv_reduction = nn.Conv2d(self.out_channels, self.in_channels, kernel_size=1)

        self.norm1 = norm_layer_dict[norm_layer](dim)

        self.attention = VHA(embed_dim=channel_dim, 
            num_heads=num_heads,
            n_win=n_win,
            topk=topk,
            side_dwconv=side_dwconv,
            kv_downsample_ratio=kv_downsample_ratio,
            kv_per_win=kv_per_win,
            qkv_bias=qkv_bias, 
            proj_bias=proj_bias
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # FFN
        self.norm2 = norm_layer_dict[norm_layer](dim)
        self.ffn = SwiGLUFFN(in_features=dim, hidden_features=int(dim * ffn_ratio),
                             bias=ffn_bias, align_to=align_to)
        
        
    def forward(self, x: torch.Tensor, pos_enc):
        B, C, D, H, W = x.shape


        x = rearrange(x, "b c d h w -> b (c d) h w")

        # attention
        x_attn = x + self.drop_path(self.inv_reduction(self.attn_dropout(self.attention(self.reduction(self.norm1(x)), pos_enc))))

        # FFN
        x_ffn = x_attn + self.drop_path(self.ffn(self.norm2(x_attn)))

        x_ffn = rearrange(x_ffn, "b (c d) h w -> b c d h w", c=C)
        return x_ffn


class DWConv_SpatialBlock(nn.Module):
    """
    Spatial Block with Depth-wise Convolution for Positional Encoding.
    
    Args:
        x: Input tensor of shape (B, C, D, H, W)
    """
    def __init__(self, channel_dim: int, 
                 num_heads: int,
                 n_win=7,                # Region partition size
                 topk=4,                 # Routing sparsity
                 side_dwconv=5,          # LEPE kernel size
                 kv_downsample_ratio=4,  # KV compression ratio
                 kv_per_win=2,           # KV per window
                 norm_layer: str = 'layernormbf16', 
                 align_to: int = 64, 
                 qkv_bias=False, 
                 proj_bias=True, 
                 ffn_bias=True, 
                 ffn_ratio: float = 4.0, 
                 drop_path: float = 0.0, 
                 dropout: float = 0.0,
                 before_attn_dwconv=3,
                 ):
        super().__init__()

        self.pos_embed = nn.Conv2d(channel_dim, channel_dim, kernel_size=before_attn_dwconv, padding=1, groups=channel_dim)
        
        # attention
        self.norm1 = norm_layer_dict[norm_layer](channel_dim)        
        self.attention = VHA(embed_dim=channel_dim, 
            num_heads=num_heads, 
            n_win=n_win,        
            topk=topk,           
            side_dwconv=side_dwconv,
            kv_downsample_ratio=kv_downsample_ratio,
            kv_per_win=kv_per_win,
            qkv_bias=qkv_bias, 
            proj_bias=proj_bias
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # FFN
        self.norm2 = norm_layer_dict[norm_layer](channel_dim)
        self.ffn = SwiGLUFFN(in_features=channel_dim, hidden_features=int(channel_dim * ffn_ratio), 
                             bias=ffn_bias, align_to=align_to)


    def forward(self, x: torch.Tensor, pos_enc=None):
        ndim = x.ndim
        if ndim == 5:
            B, C, D, H, W = x.shape
            x = rearrange(x, "b c d h w -> (b d) c h w").contiguous()
        
        x = x + self.pos_embed(x)

        # attention
        x_attn = x + self.drop_path(self.attn_dropout(self.attention(self.norm1(x), pos_enc)))

        # FFN
        x_ffn = x_attn + self.drop_path(self.ffn(self.norm2(x_attn)))

        if ndim == 5:
            x_ffn = rearrange(x_ffn, "(b d) c h w -> b c d h w", b=B).contiguous()
        return x_ffn
    