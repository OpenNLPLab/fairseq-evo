from .model import *


@register_model_architecture('lra', 'tnn_lra_listops')
def tnn_lra_listops(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'tnn')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 160)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 160)
    args.truncation_length = getattr(args, 'truncation_length', 1024)
    args.max_positions = getattr(args, 'max_positions', 2002)
    args.norm_type = getattr(args, 'norm_type', 'scalenorm')
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)
    # tnn
    args.encoder_normalize_before = True
    # gtu
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 80)
    args.toep_type = 1
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # rpe
    args.rpe_embedding = 32
    args.rpe_layers = 1
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim

@register_model_architecture('lra', 'tnn_lra_imdb')
def tnn_lra_imdb(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'tnn')
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 256)
    args.max_positions = getattr(args, 'max_positions', 4002)
    args.norm_type = getattr(args, 'norm_type', 'scalenorm')
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)
    # tnn
    args.encoder_normalize_before = True
    # gtu
    args.toep_type = 1
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # rpe
    args.rpe_embedding = 32
    args.rpe_layers = 1
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim

@register_model_architecture('lra', 'tnn_lra_aan')
def tnn_lra_aan(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'tnn')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 256)
    args.classifier_in_dim = getattr(args, 'classifier_in_dim', args.encoder_embed_dim * 2)
    args.max_positions = getattr(args, 'max_positions', 4002)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)
    # tnn
    args.encoder_normalize_before = True
    # gtu
    args.toep_type = 1
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # rpe
    args.rpe_embedding = 32
    args.rpe_layers = 1
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim

@register_model_architecture('lra', 'tnn_lra_cifar10')
def tnn_lra_cifar10(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'tnn')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 320)
    args.encoder_layers = getattr(args, 'encoder_layers', 8)
    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 160)
    args.norm_type = getattr(args, 'norm_type', 'batchnorm')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 320)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 1024)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)
    # tnn
    args.encoder_normalize_before = True
    # gtu
    args.toep_type = 1
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # rpe
    args.rpe_embedding = 32
    args.rpe_layers = 1
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim
    
@register_model_architecture('lra', 'tnn_lra_pf32')
def tnn_lra_pf32(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'tnn')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.norm_type = getattr(args, 'norm_type', 'batchnorm')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 256)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.max_positions = getattr(args, 'max_positions', 1024)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)
    # tnn
    args.encoder_normalize_before = True
    # gtu
    args.toep_type = 1
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # rpe
    args.rpe_embedding = 32
    args.rpe_layers = 1
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim

@register_model_architecture('lra', 'tnn_lra_pf128')
def tnn_lra_pf128(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'tnn')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 128)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.norm_type = getattr(args, 'norm_type', 'batchnorm')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 128)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.max_positions = getattr(args, 'max_positions', 128 * 128)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)
    # tnn
    args.encoder_normalize_before = True
    # gtu
    args.toep_type = 1
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # rpe
    args.rpe_embedding = 32
    args.rpe_layers = 1
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim
