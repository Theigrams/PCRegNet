import argparse
import logging
import os
import sys

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from data.modelnet40 import ModelNet40
from models import pcrnet
from tools import ChamferDistance
