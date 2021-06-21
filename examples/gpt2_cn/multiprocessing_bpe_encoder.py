#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
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
import argparse
import contextlib
import sys
from collections import Counter
from multiprocessing import Pool

from fairseq.data.encoders.gpt2_bpe import get_encoder

import pdb
def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab",
        help="path to encoder.json",
    )
    parser.add_argument(
        "--vocab_model",
        type=str,
        help="path to vocab.bpe",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["-"],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["-"],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument(
        "--method",
        default="sentence",
        choices=["block_size","sentence"],
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=1024)
    args = parser.parse_args()
    assert len(args.inputs) == len(
        args.outputs
    ), "number of input and output paths should match"

    if args.method == "sentence":
        with contextlib.ExitStack() as stack:
            inputs = [
                stack.enter_context(open(input, "r", encoding="utf-8"))
                if input != "-"
                else sys.stdin
                for input in args.inputs
            ]
            outputs = [
                stack.enter_context(open(output, "w", encoding="utf-8"))
                if output != "-"
                else sys.stdout
                for output in args.outputs
            ]

            encoder = MultiprocessingEncoder(args)
            pool = Pool(args.workers, initializer=encoder.initializer)
            encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)
            stats = Counter()
            for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
                if filt == "PASS":
                    for enc_line, output_h in zip(enc_lines, outputs):
                        print(enc_line, file=output_h)
                else:
                    stats["num_filtered_" + filt] += 1
                if i % 10000 == 0:
                    print("processed {} lines".format(i), file=sys.stderr)

            for k, v in stats.most_common():
                print("[{}] filtered {} lines".format(k, v), file=sys.stderr)
    elif args.method == "block_size":
        with contextlib.ExitStack() as stack:
            inputs = [
                stack.enter_context(open(input, "r", encoding="utf-8"))
                for input in args.inputs
            ]
            encoder = MultiprocessingEncoder(args)
            pool = Pool(args.workers, initializer=encoder.initializer)
            encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)
            encoded_lines_list = [line[1][0].split(' ') for line in encoded_lines]
            concatenated_lines = sum(encoded_lines_list, [])
            total_length = len(concatenated_lines)
            total_length = (total_length // args.block_size) * args.block_size
            result = [concatenated_lines[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            with open(args.outputs[0],'w') as f_write:
                for block_encoded_line in result:
                    block_encoded_line = [str(i) for i in block_encoded_line]
                    block_encoded_line = ' '.join(block_encoded_line).strip()
                    f_write.write(block_encoded_line+'\n')

class MultiprocessingEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = GPT2BPE_CN(self.args.vocab, self.args.vocab_model)

    def encode(self, line):
        global bpe
        ids_str = bpe.encode(line)
        return ids_str

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line_strip = line.strip()
            if len(line_strip) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            if len(line_strip) == 0:
                tokens = self.encode(line_strip)
                enc_lines.append(tokens)
                return ["PASS", enc_lines]
            tokens = self.encode(line)
            enc_lines.append(tokens)
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]

class GPT2BPE_CN(object):

    def __init__(self, vocab_file,model_file):
        # self.encoder = json.load(open(vocab_file))
        f = open(vocab_file,'r')
        lines = f.readlines()
        self.encoder = {}
        for line in enumerate(lines):
            key = line[1].split('\t')[0]
            self.encoder[key] = line[0]

        self.decoder = {v:k for k,v in self.encoder.items()}

        self.sp = spm.SentencePieceProcessor(model_file=model_file)
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
if __name__ == "__main__":
    main()
