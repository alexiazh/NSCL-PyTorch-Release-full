#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : desc_nscl_derender_vqs.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/18/2021
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import torch

from jacinle.utils.container import GView
from nscl.nn.reasoning_v1.quasi_symbolic import set_apply_self_mask
from nscl.models.reasoning_v1 import make_reasoning_v1_configs, ReasoningV1Model
from nscl.models.utils import canonize_monitors, update_from_loss_module

configs = make_reasoning_v1_configs()

configs.model.sg_dims = [None, 256, 256]
configs.model.vse_hidden_dims = [None, 300, 300]

configs.model.vse_known_belong = False
configs.model.vse_large_scale = True
configs.model.vse_ls_load_concept_embeddings = True
configs.train.scene_add_supervision = False
configs.train.qa_add_supervision = True

set_apply_self_mask('relate', False)


class Model(ReasoningV1Model):
    def __init__(self, vocab):
        super().__init__(vocab, configs)
        self.reasoning.embedding_attribute.concept_embeddings.weight.requires_grad = False
        self.reasoning.embedding_attribute.attribute_embeddings.weight.requires_grad = False
        self.reasoning.embedding_relation.concept_embeddings.weight.requires_grad = False

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)
        monitors, outputs = {}, {}

        self.resnet.eval()  # trigger the eval mode BatchNorm
        with torch.no_grad():
            f_scene = self.resnet(feed_dict.image)
        f_sng = self.scene_graph(f_scene, feed_dict.objects, feed_dict.objects_length)

        programs = feed_dict.program_qsseq
        programs, buffers, answers = self.reasoning(f_sng, programs, fd=feed_dict)
        outputs['answer'] = answers

        update_from_loss_module(monitors, outputs, self.qa_loss(feed_dict, answers))

        canonize_monitors(monitors)

        if self.training:
            loss = monitors['loss/qa']
            return loss, monitors, {}
        else:
            outputs['buffers'] = buffers
            outputs['monitors'] = monitors
            return outputs


def make_model(args, vocab):
    return Model(vocab)

