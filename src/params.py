import json
from logging import Logger
import torch
import os
from constants.constants import *
from src.logger import *

class Hyperparameters():
    def __init__(self, logger: Logger, checkpoint_name=None):
        self.logger = logger
        script_dir = os.path.dirname(os.path.realpath(__file__))

        path = '../checkpoints/{}/params.json'.format(checkpoint_name)
        params_path = os.path.join(script_dir, path)
        data_sources_path = os.path.join(script_dir, '../configs/data_sources.json')

        # Load params
        with open(params_path, 'r') as f:
            params = json.load(f)
        self.file_params = params
        
        # Load data sources
        with open(data_sources_path, 'r') as f:
            data_sources = json.load(f)

        self.device = torch.device(cuda if torch.cuda.is_available() else cpu)
        self.torch_seed = params[torch_seed]
        torch.manual_seed(self.torch_seed)

        self.name = params[name]
        self.data_source = params[data_source]
        self.training_data_url = data_sources[params[data_source]]

        self.ctx_length = params[ctx_length]
        self.batch_size = params[batch_size]
        self.num_dim = params[num_dim]
        self.num_head = params[num_head]
        self.num_layer = params[num_layer]
        self.head_size = self.num_dim // self.num_head
        self.dropout = params[dropout]

        self.learning_rate = params[learning_rate]

        self.num_batch = params[num_batch]
        self.eval_interval = params[eval_interval]
        self.eval_size = params[eval_size]
        self.val_split = params[val_split]
        self.test_split = params[test_split]
