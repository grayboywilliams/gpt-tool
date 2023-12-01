import json
from logging import Logger
import torch
from constants.constants import *
from src.logger import *

script_dir = os.path.dirname(os.path.realpath(__file__))

class Hyperparameters():
    def __init__(self, logger: Logger, cpt_name=None):
        self.logger = logger

        params = load_params(cpt_name)
        data_sources = load_data_sources()

        self.file_params = params
        self.device = torch.device(cuda if torch.cuda.is_available() else cpu)
        self.torch_seed = params[torch_seed]
        torch.manual_seed(self.torch_seed)

        # model
        self.name = params[name]
        self.data_source = params[data_source]
        self.training_data_url = data_sources[params[data_source]]

        # architecture params
        self.ctx_length = params[ctx_length]
        self.num_dim = params[num_dim]
        self.num_head = params[num_head]
        self.num_layer = params[num_layer]

        # training
        self.batch_size = params[batch_size]
        self.num_batch = params[num_batch]
        self.learning_rate = params[learning_rate]
        self.dropout = params[dropout]
        self.eval_interval = params[eval_interval]
        self.eval_size = params[eval_size]
        self.val_split = params[val_split]
        self.test_split = params[test_split]

def is_training_param(key):
    return key in [
        batch_size,
        num_batch,
        learning_rate,
        dropout,
        eval_interval,
        eval_size,
        val_split,
        test_split
        ]

def load_params(name=None):
    if name == None:
        name = temp

    params_path = os.path.join(script_dir, checkpoints, name, 'params.json')
    
    with open(params_path, 'r') as f:
        return json.load(f)

def load_data_sources():
    data_sources_path = os.path.join(script_dir, configs, 'data_sources.json')
    
    with open(data_sources_path, 'r') as f:
        return json.load(f)
