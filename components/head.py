import torch
import torch.nn as nn
from torch.nn import functional as F
from models.params import *
from models.dataset import *

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, params: Hyperparameters):
        super().__init__()
        self.key = nn.Linear(params.num_dim, params.head_size, bias=False)
        self.query = nn.Linear(params.num_dim, params.head_size, bias=False)
        self.value = nn.Linear(params.num_dim, params.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(params.block_size, params.block_size)))
        self.dropout = nn.Dropout(params.dropout)
        self.to(params.device)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
