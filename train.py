# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from trainer_sgd import TrainerSGD
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    if opts.use_sgd:
        trainer = TrainerSGD(opts)
    else:
        trainer = Trainer(opts)
    trainer.train()

# Examples:
# CUDA_VISIBLE_DEVICES=1 python3 train.py --model_name stereo_672x384 --use_stereo --frame_ids 0 --width 384 --height 672
#
