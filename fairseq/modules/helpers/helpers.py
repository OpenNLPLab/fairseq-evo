import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("print_config")

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def logging_info(string):
    if is_main_process():
        logger.info(string)

def print_params(**kwargs):
    if is_main_process():
        logger.info(f"start print config of {kwargs['__class__']}")
        for key in kwargs:
            if key in ["__class__", "self"]:
                continue
            logger.info(f"{key}: {kwargs[key]}")
        logger.info(f"end print config of {kwargs['__class__']}")

def print_config(config):
    if is_main_process():
        logger.info(f"start print config of {config['__class__']}")
        for key in config:
            if key in ["__class__", "self"]:
                continue
            logger.info(f"{key}: {config[key]}")
        logger.info(f"end print config of {config['__class__']}")

def get_activation_fn(activation):
    logger.info(f"activation: {activation}")
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "exp":
        return torch.exp
    elif activation == "leak":
        return F.leaky_relu
    elif activation == "1+elu":
        def f(x):
            return 1 + F.elu(x)
        return f
    elif self.act_fun == "2+elu":
            def f(x):
                return F.elu(x) + 2
            return f
    elif activation == "silu":
        return F.silu
    else:
        return lambda x: x
    
    # def get_act_fun(self):
    #     print(self.act_fun)
    #     if self.act_fun == "gelu":
    #         return F.gelu
    #     elif self.act_fun == "relu":
    #         return F.relu
    #     elif self.act_fun == "elu":
    #         return F.elu
    #     elif self.act_fun == "sigmoid":
    #         return F.sigmoid
    #     elif self.act_fun == "exp":
    #         return torch.exp
    #     elif self.act_fun == "1+elu":
    #         def f(x):
    #             return F.elu(x) + 1
    #         return f
    #     elif self.act_fun == "1+relu":
    #         def f(x):
    #             return F.relu(x) + 1
    #         return f
    #     elif self.act_fun == "2+elu":
    #         def f(x):
    #             return F.elu(x) + 2
    #         return f
    #     elif self.act_fun == "relu2":
    #         def f(x):
    #             return torch.square(torch.relu(x))
    #         return f
    #     elif self.act_fun == "leak":
    #         def f(x):
    #             return F.leaky_relu(x, negative_slope=self.negative_slope)
    #         return f
    #     else:
    #         def f(x):
    #             return x
    #         return f
