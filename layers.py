

import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt
from hparams import device




def abs_positional_encoding(max_position, d_model, n=3):
   
    # set of all positions to consider
    positions = torch.arange(max_position).float().to(device)

    # get angles to input to sinusoid functions
    k = torch.arange(d_model).float().to(device)
    coeffs = 1 / torch.pow(10000, 2 * (k // 2) / d_model)
    angles = positions.view(-1, 1) @ coeffs.view(1, -1)

    # apply sin to the even indices of angles along the last axis
    angles[:, 0::2] = torch.sin(angles[:, 0::2])

    # apply cos to the odd indices of angles along the last axis
    angles[:, 1::2] = torch.cos(angles[:, 1::2])

    return angles.view(*[1 for _ in range(n-2)], max_position, d_model)


def skew(t):
  
    # pad T
    padded = F.pad(t, [1, 0])

    # reshape to diagonalize the columns in the last 2 dimensions
    Srel = padded.reshape(-1, t.shape[-1] + 1, t.shape[-2])

    # final touches
    Srel = Srel[:, 1:]              # slice last L values
    Srel = Srel.reshape(*t.shape)   # reshape to shape of t
    return Srel


def rel_scaled_dot_prod_attention(q, k, v, e=None, mask=None):
   
    QKt = torch.matmul(q, k.transpose(-1, -2))  # (..., seq_len_q, seq_len_k)

    if e is None:
        # assumes q.shape[:-2] == k.shape[:-2]
        Srel = torch.zeros(*q.shape[:-2], q.shape[-2], k.shape[-2], device=q.device)
    else:
        Srel = skew(torch.matmul(q, e.transpose(-1, -2)))  # (..., seq_len_q, seq_len_k)

    # find and scale attention logits
    dk = sqrt(k.shape[-1])
    scaled_attention_logits = (QKt + Srel) / dk  # (..., seq_len_q, seq_len_k)

    # add scaled mask to 0 out positions to mask in softmax
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # calculate attention by calculating attention weights by softmaxing on the last dimension
    # and then multiplying by v
    return torch.matmul(F.softmax(scaled_attention_logits, dim=-1), v)


class MultiHeadAttention(nn.Module):
   
    def __init__(self, d_model, num_heads, max_rel_dist, bias=True):
       
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_rel_dist = max_rel_dist
        self.batch_first = False

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible into num_heads heads")

        self.depth = self.d_model // self.num_heads

        self.wq = nn.Linear(self.d_model, self.d_model, bias=bias)  # parameter matrix to generate Q from input
        self.wk = nn.Linear(self.d_model, self.d_model, bias=bias)  # parameter matrix to generate K from input
        self.wv = nn.Linear(self.d_model, self.d_model, bias=bias)  # parameter matrix to generate V from input

        self.E = nn.Embedding(self.max_rel_dist, self.d_model)      # relative position embeddings

        self.wo = nn.Linear(self.d_model, self.d_model, bias=True)  # final output layer

    @staticmethod
    def split_heads(x, num_heads, depth=None):
      
        # get depth if None
        if depth is None:
            if x.shape[-1] % num_heads != 0:
                raise ValueError("d_model must be divisible into num_heads")
            depth = x.shape[-1] // num_heads

        # reshape and transpose x
        x = x.view(*x.shape[:-1], num_heads, depth)     # (..., L, num_heads, depth)
        return x.transpose(-2, -3)                      # (..., num_heads, L, depth)

    def get_required_embeddings(self, seq_len, max_len=None):
       
        if max_len is None:
            max_len = self.E.num_embeddings

        # required relative position embeddings
        E_dev = self.E.weight.device
        first_emb = self.E(torch.arange(0, 1, device=E_dev)).clone()
        return torch.cat(
            [*[first_emb.clone() for _ in range(max(seq_len - max_len, 0))],
             self.E(torch.arange(max(max_len - seq_len, 0), max_len, device=E_dev))],
            dim=0
        )

    def forward(self, q, k, v, mask=None):
       
        # get Q, K, V
        q = self.wq(q) 
        k = self.wk(k)  
        v = self.wv(v)  

        # get required embeddings from E
        seq_len_k = k.shape[-2]
        e = self.get_required_embeddings(seq_len_k, self.max_rel_dist)  

       
        q = self.split_heads(q, self.num_heads, self.depth)  
        k = self.split_heads(k, self.num_heads, self.depth)  
        v = self.split_heads(v, self.num_heads, self.depth)  
        e = self.split_heads(e, self.num_heads, self.depth) 

        
        rel_scaled_attention = rel_scaled_dot_prod_attention(q, k, v, e, mask=mask)

       
        rel_scaled_attention = rel_scaled_attention.transpose(-2, -3)  
        sh = rel_scaled_attention.shape
        return self.wo(rel_scaled_attention.reshape(*sh[:-2], self.d_model))


class PointwiseFFN(nn.Module):
   
    def __init__(self, d_model, d_ff, bias=True):
        
        super(PointwiseFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.main = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=bias)
        )

    def forward(self, x):
        return self.main(x)


class DecoderLayer(nn.Module):
   
    def __init__(self, d_model, num_heads, d_ff, max_rel_dist, bias=True, dropout=0.1, layernorm_eps=1e-6):
        
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_rel_idst = max_rel_dist

        self.self_attn = MultiHeadAttention(d_model, num_heads, max_rel_dist, bias)
        self.ffn = PointwiseFFN(d_model, d_ff, bias)

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory=None, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, 
                tgt_is_causal=None, memory_is_causal=None):
       
        # multi-head attention block
        attn_out = self.layernorm1(tgt)
        attn_out = self.self_attn(attn_out, attn_out, attn_out, mask=tgt_mask)
        attn_out = self.dropout1(attn_out)
        attn_out = tgt + attn_out

        # pointwise ffn block
        ffn_out = self.layernorm2(attn_out)
        ffn_out = self.ffn(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        ffn_out = ffn_out + attn_out

        return ffn_out
