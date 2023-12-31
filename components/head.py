import torch
import torch.nn as nn
from torch.nn import functional as F
from src.params import *
from src.dataset import *

class Head(nn.Module):
    """ Single head of self-attention """

    def __init__(self, params: Hyperparameters):
        super().__init__()
        head_size = params.num_dim // params.num_head
        self.key = nn.Linear(params.num_dim, head_size, bias=False)
        self.query = nn.Linear(params.num_dim, head_size, bias=False)
        self.value = nn.Linear(params.num_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(params.ctx_length, params.ctx_length)))
        self.dropout = nn.Dropout(params.dropout)
        self.to(params.device)

    def forward(self, x):
        q = self.query(x) # (B,T,C)
        k = self.key(x)   # (B,T,C)
        v = self.value(x) # (B,T,C)

        # compute attention scores ("affinities")
        B,T,C = x.shape
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
