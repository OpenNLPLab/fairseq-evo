# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn
from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (FairseqIncrementalDecoder, FairseqLanguageModel,
                            register_model, register_model_architecture)

logger = logging.getLogger(__name__)
from typing import Dict, List, Optional

import torch
from fairseq.models.transformer import (DEFAULT_MIN_PARAMS_TO_WRAP, Embedding,
                                        TransformerDecoder)
from fairseq.models.transformer_lm import (DEFAULT_MAX_TARGET_POSITIONS,
                                           TransformerLanguageModel,
                                           TransformerLanguageModelConfig,
                                           base_lm_architecture,
                                           transformer_lm_big)
from fairseq.modules import (AdaptiveInput, CharacterTokenEmbedder,
                             MemTransformerLM)
from omegaconf import II

NoneType = None.__class__

@dataclass
class TransformerXLConfig(FairseqDataclass):
    n_token: int = 100
    n_layer: int = 4
    n_head: int = 2
    d_model: int = 200
    d_head: int = 2
    d_inner: int = 200
    dropout: float = 0.0
    dropatt: float = 0.0
    tie_weight: bool = True
    d_embed: NoneType = None
    div_val: int = 1
    tie_projs: List[bool] = field(default_factory=lambda: [False])
    pre_lnorm: NoneType = False
    tgt_len: NoneType = None
    ext_len: NoneType = None
    mem_len: NoneType = None
    cutoffs: List[int] = field(default_factory=lambda: [])
    adapt_inp: bool = False
    same_length: bool = False
    attn_type: int = 0
    clamp_len: int = -1
    sample_softmax: int = -1
    use_ada: bool = True
    use_linear: bool = False


@register_model("transformer-xl", dataclass=TransformerLanguageModelConfig)
class TransformerXL(FairseqLanguageModel):
    @classmethod
    def build_model(cls, args, task):
        return cls(TransformerXLModel(args, task))

class TransformerXLModel(FairseqIncrementalDecoder):
    def __init__(self, args, task):

        super().__init__(task.target_dictionary)

        config = TransformerXLConfig(
            n_token=len(task.target_dictionary),
            d_model=args.d_model,
            n_head=args.n_head,
            d_head=args.d_head,
            d_inner=args.d_inner,
            n_layer=args.n_layer,
            dropout=args.dropout,
            dropatt=args.dropatt,
            pre_lnorm=args.pre_lnorm,
            tgt_len=args.tgt_len,
            ext_len=args.ext_len,
            mem_len=args.mem_len,
            use_linear=args.use_linear,
            use_ada=args.use_ada,
        )
        self.config = config
        logger.info(config)
        del config.__dict__['_name']
        self.model = MemTransformerLM(**config.__dict__)
        self.cache_size = args.mem_len

        self._mems = None

    def forward(
        self,
        src_tokens,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        encoder_out=None,
        src_lengths=None,
    ):
        bsz = src_tokens.size(0)
        if incremental_state is not None:  # used during inference
            mems = self.get_incremental_state("mems")
            src_tokens = src_tokens[:, -1:]  # only keep the most recent token
        else:
            mems = self._mems

        output = self.model(
            src_tokens, 
            mems,
        )
        if incremental_state is not None:
            self.set_incremental_state(incremental_state, "mems", output[1])
        else:
            self._mems = output[1]
        return (output[0],)

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        new_order: torch.Tensor,
    ):
        """Reorder incremental state.
        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        raise NotImplementedError("This is required for generation/beam search")
    
@register_model_architecture("transformer-xl", "transformer_xl_base")
def transformer_xl_base(args):
    base_lm_architecture(args)
    args.d_model = args.decoder_embed_dim
    args.n_head = args.decoder_attention_heads
    args.d_head = args.d_model // args.n_head
    args.d_inner = args.decoder_ffn_embed_dim
    args.n_layer = args.decoder_layers
    args.dropout = args.dropout
    args.dropatt = 0.0
    args.use_linear = False
    args.pre_lnorm = True
    args.tgt_len = 0
    args.ext_len = 0
    args.mem_len = 128
    args.use_ada = False

@register_model_architecture("transformer-xl", "linear_transformer_xl_base")
def linear_transformer_xl_base(args):
    base_lm_architecture(args)
    args.d_model = args.decoder_embed_dim
    args.n_head = args.decoder_attention_heads
    args.d_head = args.d_model // args.n_head
    args.d_inner = args.decoder_ffn_embed_dim
    args.n_layer = args.decoder_layers
    args.dropout = args.dropout
    args.dropatt = 0.0
    args.use_linear = True
    args.pre_lnorm = True
    args.tgt_len = 0
    args.ext_len = 0
    args.mem_len = 128
    args.use_ada = False
