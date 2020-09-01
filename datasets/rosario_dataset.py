from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .stereo_dataset import StereoDataset



class ROSARIODataset(StereoDataset):

    def __init__(self, *args, **kwargs):
        super(ROSARIODataset, self).__init__(*args, **kwargs)

        self.K_l = np.array([[348.522264/672, 0, 344.483596/672, 0],
                             [0, 348.449870/384*1.0213, 188.808062/384*1.0213, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.float32)
        self.K_r = np.array([[349.569635/672, 0, 340.836585/672, 0],
                            [0, 349.390781/384*1.0213, 206.105440/384*1.0213, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
        self.stereo_T_l = np.array([[0.9999, 0.0012, 0.0136, -0.1188],
                             [-0.0011, 1.0, -0.0031, 0],
                             [-0.0136, 0.0031, 0.9999, -0.0003],
                             [0, 0, 0, 1.0]], dtype=np.float32)

        self.stereo_T_r = np.array([[0.9999, -0.0011, -0.0136, 0.1188],
                             [0.0012, 1.0, 0.0031, 0.0001],
                             [0.0136, -0.0031, 0.9999, 0.0019],
                             [0, 0, 0, 1.0]], dtype=np.float32)

        self.full_res_shape = (672, 384)

    def check_depth(self):
        raise NotImplementedError

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        width, height = color.size
        height = height + 8
        color.resize((width, height))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class ROSARIORAWDataset(ROSARIODataset):
    def __init__(self, *args, **kwargs):
        super(ROSARIORAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_id, side):
        if side == 'l':
            image_path = os.path.join(self.data_path, folder, 'left_' + frame_id)
        elif side == 'r':
            image_path = os.path.join(self.data_path, folder, 'right_' + frame_id)
        return image_path