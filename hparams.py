

from torch import torch, device as d
from vocabulary import vocab_size

# get device
if torch.backends.cuda.is_built():
    dev = "cuda:0"
else:
    dev = "cpu"
device = d(dev)

# default hparams for the model
hparams = {
    "d_model": 128,
    "num_layers": 3,
    "num_heads": 8,
    "d_ff": 512,
    "max_rel_dist": 1024,
    "max_abs_position": 0,
    "vocab_size": vocab_size,
    "bias": True,
    "dropout": 0.1,
    "layernorm_eps": 1e-6
}

# hparams for TF model - significantly larger
hparams_large = {
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 8,
    "d_ff": 1024,
    "max_rel_dist": 1024,
    "max_abs_position": 0,
    "vocab_size": vocab_size,
    "bias": True,
    "dropout": 0.1,
    "layernorm_eps": 1e-6
}
