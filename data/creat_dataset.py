import json
import os

import torch
import h5py

from dataloaders import ModelNet40

CATEGORY = "car"
FOLDER40 = "modelnet40_ply_hdf5_2048"
CATFILE40 = "data/modelnet40_ply_hdf5_2048/shape_names.txt"
# CATFILE10 = 'data/modelnet10_hdf5_2048/shape_names_10.txt'
NUMPOINTS = 2048
BATCHSIZE = 32

ModelNet40(
    NUMPOINTS,
    transforms=None,
    train=True,
    download=False,
    folder=FOLDER40,
    include_shapes=False,
)

all_categories = [line.rstrip() for line in open(CATFILE40)]
