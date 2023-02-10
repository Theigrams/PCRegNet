# the code is mainly adapted from https://github.com/vinits5/learning3d/blob/master/models/pointnet.py
import torch
import torch.nn as nn

from pooling import Pooling


class PointNet(nn.Module):
    def __init__(self, emb_dims=1024, in_dim=3, input_shape="bnc", use_bn=False):
        super().__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "input_shape must be 'bcn' (batch, channel, num_points) or 'bnc' (batch, num_points, channel)"
            )
        self.input_shape = input_shape
        self.emb_dims = emb_dims
        self.in_dim = in_dim
        self.use_bn = use_bn
        self.layers = self._make_layers()
        self.pooling = Pooling("max")

    def _make_layers(self):
        backbone = nn.Sequential()
        mlp = [64, 64, 64, 128, self.emb_dims]
        in_dim = self.in_dim
        for i, out_dim in enumerate(mlp):
            backbone.add_module(f"pointnet_conv_{i+1}", nn.Conv1d(in_dim, out_dim, 1))
            if self.use_bn:
                backbone.add_module(f"pointnet_bn_{i+1}", nn.BatchNorm1d(out_dim))
            backbone.add_module(f"pointnet_relu_{i+1}", nn.ReLU(inplace=True))
            in_dim = out_dim
        return backbone

    def forward(self, x):
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)
        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"input shape must be (batch, {self.in_dim}, num_points) or (batch, num_points, {self.in_dim})"
            )

        y = self.layers(x)  # [Batch x emb_dims x NumInPoints]

        # max pooling for global feature
        y = self.pooling(y)  # [Batch x emb_dims]
        return y


if __name__ == "__main__":
    import numpy as np

    x = torch.from_numpy(np.random.rand(4, 1024, 3).astype(np.float32))
    model = PointNet()
    y = model(x)
    print("Network Architecture: ")
    print(model)
    print(f"input shape: {x.shape}")
    print(f"output shape: {y.shape}")
