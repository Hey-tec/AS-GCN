# Author : Pey
# Time : 2021/4/9 20:22
# File_name : test.py

# --------- Import Model ---------#
import argparse
import numpy as np
import random
import torch
from torch.nn import Sigmoid
# --------- Sub Function ---------#

# --------- Main Function --------#
if __name__ == "__main__":
    print("Start coding...")

    a = torch.tensor([1, 2])
    b = torch.tensor([2, 1])
    print(a - b)
    c = a - b
    print(1 / (1 + np.exp(-1 * c)))







