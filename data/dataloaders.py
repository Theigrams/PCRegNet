# the code is mainly adapted from https://github.com/itailang/SampleNet/blob/master/registration/data/modelnet_loader_torch.py
import os
import sys
import json

from torch.utils.data import Dataset
import numpy as np
import h5py

import data_utils as d_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# the path of the root directory of the project
sys.path.append(BASE_DIR)


class ModelNet40(Dataset):
    """There are 40 classes in ModelNet40 dataset.
    Each class has 9840 training samples and 2468 test samples.
    Each sample is a point cloud with 2048 points.

    Args:
        num_points (int): Number of points in each point cloud
        transforms (callable): Transforms to be applied on a sample
        train (bool): If True, use the training set, otherwise use the test set
        normal_channel (bool): If True, use the normal information
        folder (str): Name of the folder that contains the dataset
        data (np.ndarray): Shape (9840, 2048, 3) or (9840, 2048, 6)
        label (np.ndarray): Shape (9840,)
        include_shapes (bool): If True, include the shape of the point cloud in the sample
    """

    def __init__(
        self,
        num_points=1024,
        transforms=None,
        train=True,
        normal_channel=False,
        download=True,
        folder="modelnet40_ply_hdf5_2048",
        url="https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
        include_shapes=False,
    ):
        super().__init__()
        self.num_points = num_points
        self.transforms = transforms
        self.train = train
        self.normal_channel = normal_channel
        self.folder = folder
        self.data_dir = os.path.join(BASE_DIR, folder)
        self.url = url
        self.include_shapes = include_shapes

        # download and unzip data
        file_name = os.path.basename(self.url)
        self.zip_file_path = os.path.join(BASE_DIR, file_name)

        if download:
            self.download_data()
        if not os.path.exists(self.data_dir):
            d_utils.unzip_data(self.zip_file_path, BASE_DIR, delete_zip_file=False)
        else:
            print("Data folder %s already exists." % (self.data_dir))

        # get the list of dataset files
        if self.train:
            files_path = os.path.join(self.data_dir, "train_files.txt")
        else:
            files_path = os.path.join(self.data_dir, "test_files.txt")
        self.files = self.get_data_files(files_path)

        # load the data
        self.data, self.label = self.load_data_file(self.files)

        # load the shapes
        self.shapes = []
        if self.include_shapes:
            N = len(self.files)
            if self.train:
                T = "train"
            else:
                T = "test"
            for n in range(N):
                json_path = os.path.join(
                    self.data_dir, f"ply_data_{T}_{n}_id2file.json"
                )
                with open(json_path, "r") as f:
                    shapes = json.load(f)
                self.shapes.append(shapes)

    def __getitem__(self, pc_idx):
        """The __getitem__ method randomly selects self.num_points points from
        the pc_idx point cloud.
        """
        point_set = self.data[pc_idx, ...]
        label = self.label[pc_idx]

        # randomly selects self.num_points points from the point cloud
        if self.num_points < point_set.shape[0]:
            pts_idx = np.arange(0, point_set.shape[0])
            np.random.shuffle(pts_idx)
            point_set = point_set[pts_idx[: self.num_points], :]
        else:
            raise ValueError(
                "num_points should be smaller than the number of points in the point cloud."
            )

        if self.transforms:
            point_set = self.transforms(point_set)

        if self.include_shapes:
            shape = self.shapes[pc_idx]
            return point_set, label, shape
        else:
            return point_set, label

    def __len__(self):
        return self.data.shape[0]

    def download_data(self):
        if os.path.exists(self.zip_file_path):
            print("Data file %s already exists." % (self.zip_file_path))
        elif os.path.exists(self.data_dir):
            print("Data folder %s already exists." % (self.data_dir))
        else:
            d_utils.download_data(self.url, BASE_DIR)

    def set_num_points(self, num_points):
        self.num_points = min(num_points, self.data.shape[1])

    def get_data_files(self, list_filename):
        with open(list_filename) as f:
            return [line.rstrip()[5:] for line in f]

    def load_data_file(self, files):
        """Reads the data from the h5 files and returns the data and labels.
        If self.normal_channel is True, the data will have shape (N, 2048, 6).
        """
        points, normals, labels = [], [], []
        for file in files:
            with h5py.File(os.path.join(BASE_DIR, file), "r") as f:
                cur_points = f["data"][:].astype(np.float32)
                cur_normals = f["normal"][:].astype(np.float32)
                cur_labels = f["label"][:].astype(np.int32)
                points.append(cur_points)
                normals.append(cur_normals)
                labels.append(cur_labels)
        points = np.concatenate(points, axis=0)
        normals = np.concatenate(normals, axis=0)
        if self.normal_channel:
            data = np.concatenate([points, normals], axis=-1).astype(np.float32)
        else:
            data = points.astype(np.float32)
        labels = np.concatenate(labels, axis=0).astype(np.int32)
        return data, labels


if __name__ == "__main__":

    import torch
    from torchvision import transforms

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            d_utils.PointcloudScale(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
        ]
    )
    dset = ModelNet40(16, train=True, transforms=transforms)
    print(dset[0][0])
    print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
