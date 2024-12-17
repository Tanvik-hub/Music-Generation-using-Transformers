

import torch
from math import sqrt
from torch import nn
from hparams import hparams
from layers import DecoderLayer, abs_positional_encoding




class MusicTransformer(nn.Module):
   
    def __init__(self,
                 d_model=hparams["d_model"],
                 num_layers=hparams["num_layers"],
                 num_heads=hparams["num_heads"],
                 d_ff=hparams["d_ff"],
                 max_rel_dist=hparams["max_rel_dist"],
                 max_abs_position=hparams["max_abs_position"],
                 vocab_size=hparams["vocab_size"],
                 bias=hparams["bias"],
                 dropout=hparams["dropout"],
                 layernorm_eps=hparams["layernorm_eps"]):
        
        super(MusicTransformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_rel_dist = max_rel_dist,
        self.max_position = max_abs_position
        self.vocab_size = vocab_size

        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = abs_positional_encoding(max_abs_position, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.decoder = nn.TransformerDecoder(
            DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_rel_dist=max_rel_dist,
                         bias=bias, dropout=dropout, layernorm_eps=layernorm_eps),
            num_layers=num_layers,
            norm=nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        )

        self.final = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
       
        # embed x according to Vaswani et. al, 2017
        x = self.input_embedding(x)
        x *= sqrt(self.d_model)

        # add absolute positional encoding if max_position > 0, and assuming max_position >> seq_len_x
        if self.max_position > 0:
            x += self.positional_encoding[:, :x.shape[-2], :]

        # input dropout
        x = self.input_dropout(x)

        # pass through decoder
        x = self.decoder(x, memory=None, tgt_mask=mask)

        # final projection to vocabulary space
        return self.final(x)
