import torch
import torch.nn as nn
from src.params import *
from src.dataset import *
from .head import *

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention """

    def __init__(self, params: Hyperparameters):
        super().__init__()
        self.heads = nn.ModuleList([Head(params) for _ in range(params.num_head)])
        self.proj = nn.Linear(params.num_dim, params.num_dim)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
