import torch.nn as nn
from src.params import *
from src.dataset import *
from .multiheadattention import *
from .feedforward import *

class Block(nn.Module):
    """ Transformer block """

    def __init__(self, params: Hyperparameters):
        super().__init__()
        self.ln1 = nn.LayerNorm(params.num_dim) # pre self-attn layer norm
        self.sa = MultiHeadAttention(params) # communication layer
        self.ln2 = nn.LayerNorm(params.num_dim) # pre ffwd layer norm
        self.ffwd = FeedFoward(params) # computation layer

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # the added x is the skip-connection
        x = x + self.ffwd(self.ln2(x)) # the added x is the skip-connection
        return x
