import json
import torch
import os

class Hyperparameters():
    def __init__(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, '../configs/params.json')

        with open(config_path, 'r') as f:
            config = json.load(f)

        self.batch_size = config['batch_size']
        self.block_size = config['block_size']

        self.num_batch = config['num_batch']
        self.num_head = config['num_head']
        self.num_layer = config['num_layer']
        self.num_dim = config['num_dim']

        self.eval_interval = config['eval_interval']
        self.eval_iters = config['eval_iters']

        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']

        self.torch_seed = config['torch_seed']
        torch.manual_seed(self.torch_seed)

        self.data_source = config['data_source']
        data_source = config['data_sources'][self.data_source]
        self.training_data_url = data_source['training_data_url']
        self.training_data_name = data_source['training_data_name']
        self.checkpoint_name = data_source['checkpoint_name']

        self.head_size = self.num_dim // self.num_head
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
