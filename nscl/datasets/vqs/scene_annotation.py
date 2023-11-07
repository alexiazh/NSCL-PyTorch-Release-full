#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : scene_annotation.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/10/2019
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import numpy as np


class AppendFullBbox(object):
    def __call__(self, image, boxes):
        assert False
        boxes = np.concatenate([
            boxes,
            np.array([[0, 0, image.width, image.height]], dtype=boxes.dtype)
        ], axis=0)
        return image, boxes
