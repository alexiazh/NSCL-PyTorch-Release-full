#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : definition.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/18/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import six
import os.path as osp

import jacinle.io as io
from jacinle.logging import get_logger
from nscl.datasets.definition import DatasetDefinitionBase
from nscl.datasets.common.scene_annotation import annotate_objects
from .program_translator import vqa_to_nscl

logger = get_logger(__file__)

__all__ = ['VQADefinition', 'build_vqa_dataset']


class VQADefinition(DatasetDefinitionBase):
    operation_signatures = [
        ('scene', [], [], 'object_set'),
        ('filter', ['concept'], ['object_set'], 'object_set'),
        ('relate', ['relational_concept'], ['object'], 'object_set'),

        ('query_ls', ['attribute'], ['object'], 'word'),
        ('query_ls_mc', ['attribute'], ['object'], 'word'),
        ('exist', [], ['object_set'], 'bool'),
        ('count', [], ['object_set'], 'integer'),
    ]

    EBD_CONCEPT_GROUPS = '<CONCEPTS>'
    EBD_RELATIONAL_CONCEPT_GROUPS = '<REL_CONCEPTS>'
    EBD_ATTRIBUTE_GROUPS = '<ATTRIBUTES>'

    extra_embeddings = [EBD_CONCEPT_GROUPS, EBD_RELATIONAL_CONCEPT_GROUPS, EBD_ATTRIBUTE_GROUPS]

    ls_attributes = None
    ls_concepts = None
    ls_relational_concepts = None
    _ls_concept_embeddings = None

    def load_concepts(self, filename, embeddings_filename=None):
        logger.critical('Load concepts from: "{}".'.format(filename))
        concepts = io.load(filename)
        self.ls_attributes = list()
        self.ls_concepts = list()
        self.ls_relational_concepts = list()

        all_concepts = set()
        for k, vs in concepts['answer_set'].items():
            if k.startswith('query'):
                for v in vs:
                    all_concepts.add(v.replace(' ', '_'))
        for v in concepts['concepts']:
            all_concepts.add(v.replace(' ', '_'))
        self.ls_concepts = list(all_concepts)
        self.ls_relational_concepts.extend([v.replace(' ', '_') for v in concepts['relational_concepts'].keys()])
        self.ls_attributes.extend([v.replace(' ', '_') for v in concepts['query_attributes'].keys()])

        if embeddings_filename is not None:
            logger.critical('Load concept embeddings from: "{}".'.format(embeddings_filename))
            self._ls_concept_embeddings = io.load(embeddings_filename)

    def get_ls_concept_embeddings(self):
        return self._ls_concept_embeddings

    def get_image_filename(self, scene):
        image_filename = scene['image_filename']
        if 'train' in image_filename:
            image_filename = osp.join('train', image_filename)
        if 'val' in image_filename:
            image_filename = osp.join('val', image_filename)
        return image_filename

    def annotate_scene(self, scene):
        return dict()

    def annotate_objects(self, scene):
        return annotate_objects(scene)

    def annotate_question_metainfo(self, metainfo):
        return dict()

    def annotate_question(self, metainfo):
        output = dict()
        if 'multiple_choices' in metainfo:
            output['question_multiple_choices'] = [str(answer).replace(' ', '_') for answer in metainfo.multiple_choices]
        return output

    def program_to_nsclseq(self, program, question=None):
        return vqa_to_nscl(program, question.get('multiple_choices', None))

    def canonize_answer(self, answer, question_type):
        if question_type == 'exist':
            assert answer in ('yes', 'no') or type(answer) is bool
            answer = (answer == 'yes')
        elif question_type == 'count':
            assert isinstance(answer, six.string_types) and answer.isdigit()
            answer = int(answer)
        else:
            answer = str(answer).replace(' ', '_')
        return answer

    def update_collate_guide(self, collate_guide):
        collate_guide['image'] = 'padimage'
        collate_guide['multiple_choices'] = 'skip'


def build_vqa_dataset(args, configs, image_root, scenes_json, questions_json):
    import jactorch.transforms.bbox as T
    transforms = [
        T.NormalizeBbox(),
        T.Resize(configs.data.image_size),
        T.PadMultipleOf(16),
        T.DenormalizeBbox(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    image_transform = T.Compose(transforms)

    question_transform = None
    if args.training_target == 'all':
        raise NotImplementedError()

    from nscl.datasets.datasets import NSCLDataset
    dataset = NSCLDataset(
        scenes_json, questions_json,
        image_root=image_root, image_transform=image_transform,
        vocab_json=args.data_vocab_json,
        question_transform=question_transform
    )

    args.data_concepts_json = osp.join(args.data_dir, 'concepts.json')
    args.data_concept_embeddings_pkl = osp.join(args.data_dir, 'concept-embeddings.pkl') if configs.model.vse_ls_load_concept_embeddings else None
    from nscl.datasets.definition import gdef
    gdef.load_concepts(args.data_concepts_json, args.data_concept_embeddings_pkl)

    return dataset

