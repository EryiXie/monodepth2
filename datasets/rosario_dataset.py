from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset



class ROSARIODataset(MonoDataset):

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
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class ROSARIORAWDataset(ROSARIODataset):
    def __init__(self, *args, **kwargs):
        super(ROSARIORAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
