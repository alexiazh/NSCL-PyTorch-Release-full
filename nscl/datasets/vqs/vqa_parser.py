#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : vqa_parser.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/18/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.


import re
import jaclearn.nlp.sng_parser as p

__all__ = ['VQAParser', 'parse']


# https://github.com/jacobandreas/nmn2/blob/master/extra/vqa/parse.py#L10-L20
BE_FORMS="is|are|was|were|has|have|had|does|do|did|be"

WH_RES = [
    r"^what (?:is|are) the (\w+) of",
    r"^what (\w+) (is|are)",
    r"^what (\w+) of",
    r"^(what|which|where)",
    r"^(how many)",
    r"^(can|could)",
    r"^(%s)" % BE_FORMS,
]

IGN_AMODS = {'many', 'much', 'what', 'which', 'where'}
WH2QUERY = {'what': 'thing', 'which': 'thing', 'where': 'place'}


class VQAParser(object):
    def __init__(self):
        pass

    def parse(self, sentence, answer=None):
        sentence = sentence.lower()
        graph, doc = p.parse(sentence, return_doc=True)

        for wh_re in WH_RES:
            match = re.match(wh_re, sentence)
            if match is None:
                continue

            if match.group(1) == 'how many':
                op = 'count'
                return ['count'] + self.gen_trailing_descriptor(doc) + self.gen_descriptor(graph, 0)
            elif match.group(1) in ('what', 'which', 'where'):
                op = 'query[{}]'.format(WH2QUERY[match.group(1)])
                nr_entities = len(graph['entities'])
                if nr_entities == 0:
                    return [op] + self.gen_trailing_descriptor(doc) + ['filter[_interest]'] + ['scene']
                if graph['entities'][0]['span'] == match.group(1):
                    if nr_entities > 1:
                        return [op] + self.gen_descriptor(graph, 1)
                    else:
                        return [op] + self.gen_trailing_descriptor(doc) + ['filter[_interest]'] + ['scene']
                else:
                    return [op] + self.gen_trailing_descriptor(doc) + self.gen_descriptor(graph, 0)
            elif match.group(1) in BE_FORMS.split('|') or match.group(1) in ('can', 'could'):
                op = 'exist'
                nr_entities = len(graph['entities'])
                if nr_entities == 0:
                    return [op] + self.gen_trailing_descriptor(doc) + ['filter[_interest]'] + ['scene']
                return [op] + self.gen_trailing_descriptor(doc) + self.gen_descriptor(graph, 0)
            else:
                i = 0
                if i >= len(graph['entities']):
                    op = 'thing'
                else:
                    if graph['entities'][0]['lemma_head'] == 'what': i += 1
                    if i >= len(graph['entities']):
                        op = 'thing'
                    else:
                        op = graph['entities'][i]['lemma_head']; i += 1

                return ['query[{}]'.format(op)] + self.gen_trailing_descriptor(doc) + self.gen_descriptor(graph, i)
            break

        return ['query[thing]'] + self.gen_trailing_descriptor(doc) + ['filter[_interest]'] + ['scene']

    def gen_trailing_descriptor(self, doc):
        chunks = list(doc.noun_chunks)
        if len(chunks) == 0:
            return []
        head = chunks[0].root.head
        if head.lemma_ == 'be':
            return ['filter[{}]'.format(x.lemma_) for x in list(head.children)[1:] if x.pos_ == 'ADJ']
        return []

    def gen_descriptor(self, graph, entity_idx):
        if entity_idx >= len(graph['entities']):
            return ['scene']

        entity = graph['entities'][entity_idx]

        def gen():
            for e in entity['modifiers']:
                if e['dep'] == 'amod' and e['lemma_span'] not in IGN_AMODS:
                    yield 'filter[{}]'.format(e['lemma_span'])
            if entity['lemma_head'] not in IGN_AMODS:
                yield 'filter[{}]'.format(entity['lemma_head'])
        return list(gen()) + self.find_links(graph, entity_idx)

    def find_links(self, graph, entity_idx):
        for rel in reversed(graph['relations']):
            if rel['subject'] == entity_idx:
                return (
                    ['relate[{}]'.format(rel['lemma_relation'])] +
                    self.gen_descriptor(graph, rel['object'])
                )
        return ['scene']


_parser = VQAParser()


def parse(question):
    return _parser.parse(question)

