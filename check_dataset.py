from __future__ import absolute_import, division, print_function

import numpy as np
from torch.utils.data import DataLoader
from utils import *
from kitti_utils import *
import datasets


def main():
    dataset = datasets.ROSARIORAWDataset

    data_path = "../rosario"
    val_filenames = readlines(os.path.join(os.path.dirname(__file__), "splits", "rosario", "test_files.txt"))

    val_dataset = dataset(
        data_path, val_filenames, 376, 672,
        [0, 's'], 4, is_train=False)
    for batch_idx, inputs in enumerate(val_dataset):
        print(batch_idx)    
        print(inputs[('K', 0)])
        print(inputs[("stereo_T")])
        print()


if __name__ == "__main__":
    main()