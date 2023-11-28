import json
import torch
import os
from constants.constants import *

class Hyperparameters():
    def __init__(self, checkpoint_name=None):
        script_dir = os.path.dirname(os.path.realpath(__file__))

        if checkpoint_name is not None:
            path = '../checkpoints/{}/params.json'.format(checkpoint_name)
        else:
            path = '../configs/params.json'

        config_path = os.path.join(script_dir, path)
        data_sources_path = os.path.join(script_dir, '../configs/data_sources.json')

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.config = config
        
        # Load data sources
        with open(data_sources_path, 'r') as f:
            data_sources = json.load(f)

        self.device = torch.device(cuda if torch.cuda.is_available() else cpu)
        self.torch_seed = config[torch_seed]
        torch.manual_seed(self.torch_seed)

        self.name = config[name]
        self.data_source = config[data_source]
        self.training_data_url = data_sources[config[data_source]]

        self.ctx_length = config[ctx_length]
        self.batch_size = config[batch_size]
        self.num_dim = config[num_dim]
        self.num_head = config[num_head]
        self.num_layer = config[num_layer]
        self.head_size = self.num_dim // self.num_head
        self.dropout = config[dropout]

        self.learning_rate = config[learning_rate]

        self.num_batch = config[num_batch]
        self.eval_interval = config[eval_interval]
        self.eval_iters = config[eval_iters]
        self.val_split = config[val_split]
        self.test_split = config[test_split]

    def save_config(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, '../checkpoints/{}/params.json'.format(self.name))
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
