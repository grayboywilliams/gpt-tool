import torch
import urllib.request
from .params import *

class Dataset():
    def __init__(self, params: Hyperparameters, get_data=False):
        self.params = params
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.filename = os.path.join(script_dir, '../training_data', params.data_source + ".txt")
        self.source_url = params.training_data_url

        self.data = self.get_data(get_data)
        self.tokens = self.tokenize_data()
        self.vocab_size = len(self.tokens)
        self.encode = self.get_encoder()
        self.decode = self.get_decoder()
        self.train_data, self.val_data, self.test_data = self.split_data(self.params.val_split, self.params.test_split)

    def get_data(self, get_data=False):
        if get_data or os.path.isfile(self.filename) == False:
            urllib.request.urlretrieve(self.source_url, self.filename)
        with open(self.filename, 'r', encoding='utf-8') as f:
            data = f.read()
        return data

    def tokenize_data(self):
        return sorted(list(set(self.data)))

    def get_encoder(self):
        stoi = { ch:i for i,ch in enumerate(self.tokens) }
        return lambda s: [stoi[c] for c in s]
    
    def get_decoder(self):
        itos = { i:ch for i,ch in enumerate(self.tokens) }
        return lambda l: ''.join([itos[i] for i in l]) 

    def split_data(self, val_split, test_split):
        data = torch.tensor(self.encode(self.data), dtype=torch.long)
        v = int((1-(val_split+test_split))*len(data))
        t = int((1-test_split)*len(data))
        train_data = data[:v]
        val_data = data[v:t]
        test_data = data[t:]
        return train_data, val_data, test_data

    def get_batch(self, stage='train'):
        data = self.train_data if stage == 'train' else self.val_data if stage == 'val' else self.test_data
        ix = torch.randint(len(data) - self.params.ctx_length, (self.params.batch_size,))
        x = torch.stack([data[i:i+self.params.ctx_length] for i in ix])
        y = torch.stack([data[i+1:i+self.params.ctx_length+1] for i in ix])
        x, y = x.to(self.params.device), y.to(self.params.device)
        return x, y

