import argparse
import logging
import os
import sys

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from configs import ConfigLoader
from data import ModelNet40
from models import pcrnet


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_environment(args):
    pass


def main():
    pass


if __name__ == "__main__":
    main()
