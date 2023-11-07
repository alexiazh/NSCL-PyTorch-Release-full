#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/29/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

from nscl.datasets.factory import register_dataset
from .definition import VQADefinition
from .definition import build_vqa_dataset


for dataset_name in ['vqs']:
    register_dataset( dataset_name, VQADefinition, builder=build_vqa_dataset )
