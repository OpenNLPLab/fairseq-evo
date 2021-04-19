#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
        "--encoder-json",
        help="path to encoder.json",
    )
    parser.add_argument(
        "--vocab-bpe",
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
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

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
                enc_lines.append(" ".join(tokens))
                return ["PASS", enc_lines]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
