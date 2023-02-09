import torch
import torch.nn as nn
import torch.nn.functional as F


class Pooling(nn.Module):
    def __init__(self, pool_type="max"):
        super().__init__()
        self.pool_type = pool_type

    def forward(self, x):
        if self.pool_type == "max":
            return torch.max(x, 2)[0].contiguous()
        elif self.pool_type == "avg":
            return torch.mean(x, 2).contiguous()
        else:
            raise ValueError("pool_type must be 'max' or 'avg'")
