#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : program_translator.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/18/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

__all__ = ['vqa_to_nscl']


def get_vqa_op_param(op):
    return op[op.find('[') + 1:-1].replace(' ', '_')


def vqa_to_nscl(mc_program, multiple_choices=None):
    nscl_program = list()
    mapping = dict()

    if multiple_choices is not None:
        multiple_choices = [str(answer).replace(' ', '_') for answer in multiple_choices]

    last_op_idx = None
    for block_id, op in enumerate(reversed(mc_program)):
        current = None
        if op == 'scene':
            current = dict(op='scene')
            if block_id != 0:
                last_op_idx = block_id - 1
        elif op.startswith('filter'):
            concept = get_vqa_op_param(op)
            last = nscl_program[mapping[block_id - 1]]
            if last['op'] == 'filter':
                last['concept'].append(concept)
            else:
                current = dict(op='filter', concept=[concept])
        elif op.startswith('relate'):
            concept = get_vqa_op_param(op)
            current = dict(op='relate', relational_concept=[concept])
        else:
            if op.startswith('query'):
                if block_id == len(mc_program) - 1:
                    attribute = get_vqa_op_param(op)
                    if multiple_choices is None:
                        current = dict(op='query_ls', attribute=attribute)
                    else:
                        current = dict(op='query_ls_mc', attribute=attribute, multiple_choices=multiple_choices)
            elif op == 'exist':
                current = dict(op='exist')
            elif op == 'count':
                if multiple_choices is None:
                    current = dict(op='count')
                else:
                    current = dict(op='count', multiple_choices=multiple_choices)
            else:
                raise ValueError('Unknown VQA operation: {}.'.format(op))

        if current is None:
            mapping[block_id] = mapping[block_id - 1]
        else:
            if op == 'scene':
                current['inputs'] = list()
            else:
                current['inputs'] = list(map(mapping.get, [block_id - 1]))

            nscl_program.append(current)
            mapping[block_id] = len(nscl_program) - 1

    assert last_op_idx is None

    return nscl_program
