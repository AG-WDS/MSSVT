from torch import nn
import torch

class BandMulGroupSplitter2D3D(nn.Module):
    def __init__(self, split_scheme: dict = None):
        super().__init__()
        self.split_scheme = split_scheme
        self.ndim = {}
        for name, (idxs, dim) in self.split_scheme.items():
            indices_list = [torch.arange(start, end) for (start, end) in idxs]
            indices_tensor = torch.cat(indices_list)
            self.register_buffer(f"_indices_{name}", indices_tensor)
            self.ndim[name] = dim
        
    def forward(self, x):
        groups = {}
        for name in self.split_scheme.keys():
            idxs = self.get_buffer(f"_indices_{name}")
            groups[name] = torch.index_select(x, dim=2, index=idxs) if self.ndim[name] == '3D' else torch.index_select(x, dim=2, index=idxs).squeeze(1)
                    
        return groups
    
