import torch.nn as nn
from models.params import *
from models.dataset import *
from .multiheadattention import *
from .feedforward import *

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, params: Hyperparameters):
        super().__init__()
        self.sa = MultiHeadAttention(params) # communication layer
        self.ffwd = FeedFoward(params) # computation layer
        self.ln1 = nn.LayerNorm(params.num_dim) # pre layer normalization (pre sa)
        self.ln2 = nn.LayerNorm(params.num_dim) # pre layer normalization (pre ffwd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # the 'x +' is the skip-connection (residual connection)
        x = x + self.ffwd(self.ln2(x)) # the 'x +' is the skip-connection (residual connection)
        return x
