import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet import PointNet


class PCRNet(nn.Module):
    def __init__(self, emb_dims=1024, in_dim=3, input_shape="bnc"):
        super().__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "input_shape must be 'bcn' (batch, channel, num_points) or 'bnc' (batch, num_points, channel)"
            )
        self.input_shape = input_shape

        self.feat = PointNet(emb_dims=emb_dims, in_dim=in_dim, input_shape="bcn")

        self.fc1 = nn.Linear(emb_dims * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 7)

    def forward(self, x, y):
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)
            y = y.permute(0, 2, 1)

        x = self.feat(x)
        y = self.feat(y)

        z = torch.cat([x, y], dim=1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        z = self.fc6(z)  # [Batch x 7]

        pre_normalized_quat = z[:, :4]
        normalized_quat = F.normalize(pre_normalized_quat, dim=1)
        translation = z[:, 4:]
        z = torch.cat([normalized_quat, translation], dim=1)

        # returned z is a vector of 7 twist parameters.
        # pre_normalized_quat is used for loss on norm
        return z, pre_normalized_quat


if __name__ == "__main__":
    import numpy as np

    x = torch.from_numpy(np.random.rand(4, 1024, 3).astype(np.float32))
    y = torch.from_numpy(np.random.rand(4, 1024, 3).astype(np.float32))
    model = PCRNet()
    z, pre_normalized_quat = model(x, y)
    print("Network Architecture: ")
    print(model)
    print(f"input shape: {x.shape}, {y.shape}")
    print(f"output shape: {z.shape}, {pre_normalized_quat.shape}")
