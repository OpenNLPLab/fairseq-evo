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
import torch
from fairseq.data.encoders.gpt2_bpe import get_encoder
from tqdm import tqdm
import pdb
def init_distributed_mode():
    local_rank = int(os.environ['SLURM_PROCID'])
    port = "15323"
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    os.environ['LOCAL_RANK'] = str(local_rank)
    print('| distributed init (rank {})'.format(local_rank), flush=True)

#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
        type=str,
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        type=str,
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
    parser.add_argument(
        "--writen_files_dir",
        type=str,
        default="./new_1/writen_files",
        help="input files to filter/encode",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=1024)
    args = parser.parse_args()
    init_distributed_mode()
    # import pdb;pdb.set_trace()
    rank = os.environ['RANK']
    world_size = os.environ['WORLD_SIZE']
    files_all = []
    if os.path.isfile(args.inputs):
        with open(args.inputs,'r') as f:
            for line in f:
                line = line.strip()
                files_all.append(line)
    if os.path.isdir(args.inputs):
        for root,file_dir, files in os.walk(args.inputs):
            for file in files:
                filepath = os.path.join(root,file)
                files_all.append(filepath)
    files_all.sort()
    steps = len(files_all)//int(world_size)
    rank = int(rank)
    world_size = int(world_size)
    if rank < world_size-1:
        files_inputs = files_all[rank*steps:steps*(rank+1)]
    else:
        files_inputs = files_all[rank*steps:]
    files_output = args.outputs.split('.')[0]+'_'+str(rank)+'.bpe'
    start_files = 0
    os.makedirs(args.writen_files_dir, exist_ok=True)
    ## check the writen files
    if os.path.exists(os.path.join(args.writen_files_dir,str(rank)+'.bpe')):
        with open(os.path.join(args.writen_files_dir,str(rank)+'.bpe'),'r') as f:
            writen_files = f.readlines()
            num_writen_files = len(writen_files)
            start_files = num_writen_files-2
    files_inputs_ = files_inputs[start_files:]
    ##check filesize
    files_inputs = []
    for files_input in files_inputs_:
        filesize = os.path.getsize(files_input)
        filesize = filesize/float(1024*1024)
        #if filesize>300:
        #    with open(os.path.join(args.writen_files_dir, 'largefile_'+str(rank) + '.bpe'), 'a+') as f:
        #        f.write(files_input + '\n')
        #else:
        files_inputs.append(files_input)
    encoder = MultiprocessingEncoder(args)
    encoder.initializer()
    if args.method == "sentence":
        for i, input_i in enumerate(files_inputs):
            print(f'rank:{rank}_{i}/{len(files_inputs)}:{input_i}', flush=True)
            with open(files_output,'a+') as outputs:
                with open(input_i,'r', encoding="utf-8") as f_i:
                    for line in f_i:
                        encoded_line = encoder.encode_lines([line])
                        outputs.write(encoded_line[1][0]+'\n')

        '''
        sd
        with contextlib.ExitStack() as stack:
            inputs = [
                stack.enter_context(open(input, "r", encoding="utf-8"))
                if input != "-"
                else sys.stdin
                for input in files_inputs
            ]
            outputs = stack.enter_context(open(files_output, "a+", encoding="utf-8"))
            for i,input_i in enumerate(inputs):
                print(f'rank:{rank}_{i}/{len(inputs)}:{input_i}', flush=True)
                inputs_tmp = [input_i]

                encoder = MultiprocessingEncoder(args)
                pool = Pool(args.workers, initializer=encoder.initializer)
                encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs_tmp), 100)
                #pdb.set_trace()
                stats = Counter()
                for j, (filt, enc_lines) in enumerate(encoded_lines, start=1):
                    if filt == "PASS":
                        for enc_line in enc_lines:
                            print(enc_line, file=outputs,flush=True)
                    else:
                        stats["num_filtered_" + filt] += 1
                with open(os.path.join(args.writen_files_dir,str(rank)+'.bpe'),'a+') as f:
                    f.write(files_inputs[i]+'\n')
        '''
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

