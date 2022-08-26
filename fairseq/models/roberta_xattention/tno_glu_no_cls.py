import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder
from fairseq.modules import LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.roberta import RobertaEncoder, RobertaModel, base_architecture

from fairseq.models.xformer import TNOGLUEncoder
from .tno_glu import RobertaTNOGLUEncoder

class RobertaTNOGLUNoClsEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TNOGLUEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder
    
    # def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
    #     # rm cls
    #     print("=========")
    #     print(src_tokens.shape)
    #     src_tokens = src_tokens[:, 1:]
    #     print(src_tokens.shape)
    #     encoder_out = self.sentence_encoder(
    #         src_tokens,
    #         return_all_hiddens=return_all_hiddens,
    #         token_embeddings=kwargs.get("token_embeddings", None),
    #     )
    #     # T x B x C -> B x T x C
    #     features = encoder_out["encoder_out"][0].transpose(0, 1)
    #     print(features.shape)
    #     inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
    #     return features, {"inner_states": inner_states}


@register_model("roberta_tno_glu_no_cls")
class RobertaTNOGLUNoCls(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaTNOGLUNoClsEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaClassificationNoClsHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )
        
class RobertaClassificationNoClsHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features.mean(dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


##### standard pos
@register_model_architecture("roberta_tno_glu_no_cls", "roberta_tno_no_exp_base_3_1_standard_no_pos_no_cls")
def roberta_tno_no_exp_base_3_1_standard_no_pos_no_cls(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.encoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
      
@register_model_architecture("roberta_tno_glu_no_cls", "roberta_tno_no_exp_base_2_2_standard_no_pos_no_cls")
def roberta_tno_no_exp_base_2_2_standard_no_pos_no_cls(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.encoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("roberta_tno_glu_no_cls", "roberta_tno_no_exp_base_e4_s4_standard_no_pos_no_cls")
def roberta_tno_no_exp_base_e4_s4_standard_no_pos_no_cls(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 4
    args.shrink_ratio = 4
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.encoder_embed_dim // 2
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.encoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### standard pos

##### dpb 64 l6
@register_model_architecture("roberta_tno_glu_no_cls", "roberta_tno_no_exp_base_3_1_standard_no_pos_one_head_dpb_v8_64_l6_no_cls")
def roberta_tno_no_exp_base_3_1_standard_no_pos_one_head_dpb_v8_64_l6_no_cls(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 8 
    args.dpb_embedding = 64
    args.dpb_use_pad = False
    args.dpb_layers = 6
    args.residual = False
    # pos
    args.no_token_positional_embeddings = True
      
@register_model_architecture("roberta_tno_glu_no_cls", "roberta_tno_no_exp_base_2_2_standard_no_pos_one_head_dpb_v8_64_l6_no_cls")
def roberta_tno_no_exp_base_2_2_standard_no_pos_one_head_dpb_v8_64_l6_no_cls(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.encoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 8 
    args.dpb_embedding = 64
    args.dpb_use_pad = False
    args.dpb_layers = 6
    args.residual = False
    # pos
    args.no_token_positional_embeddings = True
##### dpb 64 l6

##### dpb 64 l6 decay
@register_model_architecture("roberta_tno_glu_no_cls", "roberta_tno_no_exp_base_3_1_standard_no_pos_one_head_dpb_v8_64_l6_decay_99_no_cls")
def roberta_tno_no_exp_base_3_1_standard_no_pos_one_head_dpb_v8_64_l6_decay_99_no_cls(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 8 
    args.dpb_embedding = 64
    args.dpb_use_pad = False
    args.dpb_layers = 6
    args.residual = False
    # pos
    args.no_token_positional_embeddings = True
      
@register_model_architecture("roberta_tno_glu_no_cls", "roberta_tno_no_exp_base_2_2_standard_no_pos_one_head_dpb_v8_64_l6_decay_99_no_cls")
def roberta_tno_no_exp_base_2_2_standard_no_pos_one_head_dpb_v8_64_l6_decay_99_no_cls(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 8 
    args.dpb_embedding = 64
    args.dpb_use_pad = False
    args.dpb_layers = 6
    args.residual = False
    # pos
    args.no_token_positional_embeddings = True
##### dpb 64 l6 decay