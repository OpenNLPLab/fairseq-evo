# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for OpenAI GPT."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import json
import logging
import os
import re
from io import open
import sentencepiece as spm
import jieba
from dataclasses import dataclass, field

from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass

current_path = os.getcwd()
DEFAULT_VOCAB = os.path.join(current_path,'gpt2_tokenizer_cn/vocab.vocab')
DEFAULT_VOCAB_MODEL = os.path.join(current_path,'gpt2_tokenizer_cn/vocab.model')


@dataclass
class GPT2BPECNConfig(FairseqDataclass):
    vocab_file: str = field(
        default=DEFAULT_VOCAB, metadata={"help": "path to vocab.vocab"}
    )
    model_file: str = field(
        default=DEFAULT_VOCAB_MODEL, metadata={"help": "path to vocab.model"}
    )


@register_bpe("gpt2_cn", dataclass=GPT2BPECNConfig)
class GPT2BPE_CN(object):

    def __init__(self, cfg):
        # self.encoder = json.load(open(vocab_file))
        f = open(cfg.vocab_file,'r')
        lines = f.readlines()
        self.encoder = {}
        for line in enumerate(lines):
            key = line[1].split('\t')[0]
            self.encoder[key] = line[0]

        self.decoder = {v:k for k,v in self.encoder.items()}

        self.sp = spm.SentencePieceProcessor(model_file=cfg.model_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")

    def tokenize(self, text):
        """ Tokenize a string. """
        seg_list = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        new_seg = " ".join(seg_list)
        return self.sp.encode(new_seg)


    def encode(self, text):
        res = self.tokenize(text)
        return " ".join(map(str, res))

    def decode(self, tokens):
        tokens = [int(tok) if tok not in {"<unk>", "<mask>"} else tok for tok in tokens.split()]
        text = self.sp.decode(tokens)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text

