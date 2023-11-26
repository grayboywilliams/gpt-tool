import torch.nn as nn
from models.params import *
from models.dataset import *

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, params: Hyperparameters):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(params.num_dim, 4 * params.num_dim),
            nn.ReLU(),
            nn.Linear(4 * params.num_dim, params.num_dim),
            nn.Dropout(params.dropout),
        )

    def forward(self, x):
        return self.net(x)
