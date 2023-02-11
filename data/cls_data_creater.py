import os
import sys
import json

import torch
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from data.modelnet40 import ModelNet40

CATEGORY = "car"
FOLDER40 = "modelnet40_ply_hdf5_2048"
CATFILE40 = "data/modelnet40_ply_hdf5_2048/shape_names.txt"
# CATFILE10 = 'data/modelnet10_hdf5_2048/shape_names_10.txt'
NUMPOINTS = 2048
BATCHSIZE = 32


def save_h5(h5_filename, data, label, data_dtype="uint8", label_dtype="uint8"):
    h5_fout = h5py.File(h5_filename, "w")
    h5_fout.create_dataset(
        "data", data=data, compression="gzip", compression_opts=4, dtype=data_dtype
    )
    h5_fout.create_dataset(
        "label", data=label, compression="gzip", compression_opts=1, dtype=label_dtype
    )
    h5_fout.close()


# load the dataset
ModelNet40(
    NUMPOINTS,
    transforms=None,
    train=True,
    download=False,
    folder=FOLDER40,
    include_shapes=False,
)


all_categories = [line.rstrip("\n") for line in open(CATFILE40)]
# allowed_catagories = [line.rstrip('\n') for line in open(CATFILE10)]
allowed_catagories = [CATEGORY]

# load the dataset
for T, b in zip(("train", "test"), (True, False)):
    dataset = ModelNet40(
        NUMPOINTS,
        transforms=None,
        train=b,
        download=False,
        folder=FOLDER40,
        include_shapes=True,
    )

    # filter correct classes
    newset = []
    for x in dataset:
        # print(f"label: {all_categories[x[1]]}")
        if all_categories[x[1]] in allowed_catagories:
            newset.append(x)
    loader = torch.utils.data.DataLoader(newset, batch_size=BATCHSIZE, shuffle=True)

    try:
        flist = open(f"data/{CATEGORY}_hdf5_2048/{T}_files.txt", "w")
    except FileNotFoundError:
        os.mkdir(f"data/{CATEGORY}_hdf5_2048/")
        flist = open(f"data/{CATEGORY}_hdf5_2048/{T}_files.txt", "w")

    for n, databatch in enumerate(loader):
        h5name = f"data/{CATEGORY}_hdf5_2048/{T}{n}.h5"
        jname = f"data/{CATEGORY}_hdf5_2048/ply_data_{T}_{n}_id2file.json"

        data, label, model = databatch

        save_h5(h5name, data, label, data_dtype="float32", label_dtype="uint8")
        with open(jname, "w") as write_file:
            json.dump(model, write_file)
        flist.write(f"{h5name}\n")
        print(f"saved {h5name}")

    flist.close()
