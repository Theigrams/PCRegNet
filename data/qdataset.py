import sys
from os.path import abspath, join, dirname

sys.path.insert(0, join(abspath(dirname(__file__)), ".."))

import torch
import numpy as np
import kornia.geometry.conversions as C  # works with (x, y, z, w) quaternions
import kornia.geometry.linalg as L


import tools.quaternion as Q  # works with (w, x, y, z) quaternions


def deg2rad(deg):
    return deg * np.pi / 180


def rad2deg(rad):
    return rad * 180 / np.pi


class QuaternionTransform(object):
    def __init__(self, vec: torch.Tensor, inverse: bool = False):
        # inverse: if True, first apply translation, then rotation
        self._inversion = torch.tensor([inverse], dtype=torch.bool)
        # a B x 7 vector of 4 quaternions and 3 translation parameters
        self.vec = vec.view([-1, 7])

    @staticmethod
    def from_dict(d, device):
        """Dict constructor: create a quaternion transform from a dict."""
        vec = torch.tensor(d["vec"], device=device)
        inversion = d["inversion"][0].item()
        return QuaternionTransform(vec, inversion)

    def inverse(self):
        """Inverse constructor: get the inverse of a quaternion transform."""
        quat = self.quat()
        trans = self.trans()
        quat_inv = Q.qinv(quat)
        trans_inv = -trans

        vec = torch.cat([quat_inv, trans_inv], dim=1)
        return QuaternionTransform(vec, not self.inversion())

    def as_dict(self):
        """Convert the quaternion transform to a dict."""
        return {"vec": self.vec, "inversion": self._inversion}

    def quat(self):
        """Get the quaternion part of the quaternion transform."""
        return self.vec[:, :4]

    def trans(self):
        """Get the translation part of the quaternion transform."""
        return self.vec[:, 4:]

    def inversion(self):
        """Get the inversion flag of the quaternion transform.
        To handle dataloader batching of samples, we take the first item's 'inversion' as the inversion set for the entire batch.
        """
        return self._inversion.item[0].item()

    @staticmethod
    def wxyz2xyzw(q):
        """Convert a quaternion from wxyz to xyzw format."""
        q = q[..., [1, 2, 3, 0]]
        return q

    @staticmethod
    def xyzw2wxyz(q):
        """Convert a quaternion from xyzw to wxyz format."""
        q = q[..., [3, 0, 1, 2]]
        return q

    def compute_errors(self, other):
        """Calculate Quaternion Difference Norm Error.
        http://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf chapter 3.3
        norm_err = torch.mean(
            torch.min(
                torch.sum((self.quat() - other.quat()) ** 2, dim=1),
                torch.sum((self.quat() + other.quat()) ** 2, dim=1),
            )
        )
        """
        q1 = self.quat()
        q2 = other.quat()
        R1 = C.quaternion_to_rotation_matrix(self.wxyz2xyzw(q1))
        R2 = C.quaternion_to_rotation_matrix(self.wxyz2xyzw(q2))
        R2inv = R2.transpose(1, 2)
        R1_R2inv = torch.bmm(R1, R2inv)

        # Calculate rotation error
        # rot_err = torch.norm(C.rotation_matrix_to_angle_axis(R1_R2inv), dim=1)
        # rot_err = torch.mean(rot_err)

        # Taken from PCN: Point Completion Network
        # https://arxiv.org/pdf/1808.00671.pdf
        rot_err = torch.mean(2 * torch.acos(2 * torch.sum(q1 * q2, dim=1) ** 2 - 1))

        # Calculate deviation from Identity
        batch_size = R1_R2inv.shape[0]
        I = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(R1_R2inv.device)
        norm_err = torch.mean(torch.sum((R1_R2inv - I) ** 2, dim=(1, 2)))

        trans_err = torch.mean(torch.sqrt((self.trans() - other.trans()) ** 2))

        return rot_err, norm_err, trans_err

    def rotate(self, p: torch.Tensor):
        """Rotate batch of points p by the quaternion transform."""
        ndim = p.dim()
        if ndim == 2:
            N = p.shape[0]
            assert self.vec.shape[0] == 1
            # repeat transformation vector for each point in shape
            quat = self.quat().expand([N, -1])
            p_rotated = Q.qrot(quat, p)
        elif ndim == 3:
            B, N, _ = p.shape
            quat = self.quat().expand([B, -1])
            quat = self.quat().unsqueeze(1).expand([-1, N, -1]).contiguous()
            p_rotated = Q.qrot(quat, p)

        return p_rotated


def create_random_transform(dtype, max_rotation_deg, max_translation):
    max_rotation = deg2rad(max_rotation_deg)
    rot = np.random.uniform(-max_rotation, max_rotation, size=[1, 3])
    trans = np.random.uniform(-max_translation, max_translation, size=[1, 3])
    quat = Q.euler_to_quaternion(rot, order="xyz")

    vec = np.concatenate([quat, trans], axis=1)
    vec = torch.tensor(vec, dtype=dtype)
    return QuaternionTransform(vec)


class QuaternionFixedDataset(torch.utils.data.Dataset):
    def __init__(self, data, repeat=1, seed=0, apply_noise=False, fixed_noise=False):
        self.data = data
        self.include_shapes = data.include_shapes
        self.len_data = len(data)
        self.len_set = self.len_data * repeat

        # Fix numpy seed and crtate fixed transform list
        np.random.seed(seed)
        self.transforms = [
            create_random_transform(torch.float32, 45, 0) for _ in range(self.len_set)
        ]

        self.noise = None
        if fixed_noise:
            self.noise = torch.tensor(
                [0.04 * np.random.randn(1024, 3) for _ in range(self.len_set)],
                dtype=torch.float32,
            )

        self.apply_noise = apply_noise
        self.fixed_noise = fixed_noise

    def __len__(self):
        return self.len_set

    def __getitem__(self, index):
        if self.include_shapes:
            points, _, shape = self.data[index % self.len_data]
        else:
            points, _ = self.data[index % self.len_data]
        gt_transform = self.transforms[index]
        points_rotated = gt_transform.rotate(points)
        if self.apply_noise:
            if self.fixed_noise:
                noise = self.noise[index].to(points.device)
            else:
                noise = torch.tensor(
                    0.04 * np.random.randn(1024, 3), dtype=torch.float32
                ).to(points.device)
            points_rotated += noise

        gt_dict = gt_transform.as_dict()  # points ---> points_rotated
        if self.include_shapes:
            return points, points_rotated, gt_dict, shape
        else:
            return points, points_rotated, gt_dict


if __name__ == "__main__":
    from data.data_loaders import ModelNet40
    from data.data_utils import PointcloudToTensor
    import torchvision

    transforms = torchvision.transforms.Compose([PointcloudToTensor()])
    dset = ModelNet40(
        num_points=1024, train=True, transforms=transforms, download=False
    )

    qdata = QuaternionFixedDataset(dset, repeat=2, seed=0, apply_noise=False)
    print(f"Length of dataset: {len(qdata)}")
    points, points_rotated, gt_dict = qdata[0]
    print(f"points: {points}")
    print(f"points_rotated: {points_rotated}")
    print(f"gt_dict: {gt_dict}")
