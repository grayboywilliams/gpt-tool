from logging import Logger
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from .params import *
from .dataset import *
from components.block import *
from src.logger import *
from constants.constants import *

class GPTLanguageModel(nn.Module):
    """ GPT language model """

    def __init__(self, logger: Logger, params: Hyperparameters, dataset: Dataset):
        super().__init__()
        self.logger = logger
        self.params = params
        self.dataset = dataset

        self.token_embedding = nn.Embedding(self.dataset.vocab_size, self.params.num_dim)
        self.position_embedding = nn.Embedding(params.ctx_length, self.params.num_dim)
        self.blocks = nn.Sequential(*[Block(self.params) for _ in range(self.params.num_layer)])
        self.layernorm = nn.LayerNorm(self.params.num_dim)
        self.linear = nn.Linear(self.params.num_dim, self.dataset.vocab_size)
        self.apply(self._init_weights)
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.params.learning_rate)
        self.to(self.params.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, inputs, targets=None):
        _,T = inputs.shape

        # inputs and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding(inputs) # (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=self.params.device)) # (T,C)
        x = token_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.layernorm(x) # (B,T,C)
        logits = self.linear(x) # (B,T,vocab_size)

        if targets is None:
            loss = None # is inference
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # does softmax multinomial to logits internally

        return logits, loss
    
    def begin_train(self):
        self.train()
        start_time = time.time()
        self.logger.log(SUMMARY, f"num_batch: {self.params.num_batch}, " +
                        f"eval_interval: {self.params.eval_interval}, " +
                        f"eval_size: {self.params.eval_size}")

        for iter in range(self.params.num_batch):
            # every eval_interval steps evaluate the loss on train and val sets
            if iter % self.params.eval_interval == 0 or iter == self.params.num_batch - 1:
                losses = self.estimate_loss()
                elapsed_time = time.time() - start_time
                elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                self.logger.log(SUMMARY, f"step {iter}: train loss {losses[train]:.4f}, " +
                           f"val loss {losses[val]:.4f}, time {elapsed_time_str}")

            # sample a batch of data
            inputs, targets = self.dataset.get_batch('train')

            # evaluate the loss
            _, loss = self(inputs, targets)

            # backpropogate the loss
            self.optimizer.zero_grad(set_to_none=True) # reset gradients
            loss.backward() # compute gradients
            self.optimizer.step() # update params
        
        losses = self.estimate_loss()
        preds = self.params.num_batch * self.params.batch_size
        
        self.logger.log(SUMMARY, f"final: test loss {losses[test]:.4f}")
        self.logger.log(SUMMARY, f"tokens observed: {preds * self.params.ctx_length}")
        self.logger.log(SUMMARY, f"tokens predicted: {preds}")
        self.logger.log(SUMMARY, f"effective epochs: {preds * self.params.ctx_length / self.dataset.data_size:.2f}")

    @torch.no_grad()
    def estimate_loss(self):
        mode = self.training
        self.eval()

        out = {}
        for stage in [train, val, test]:
            # get eval_size losses and average them
            losses = torch.zeros(self.params.eval_size)
            for k in range(self.params.eval_size):
                inputs, targets = self.dataset.get_batch(stage)
                _, loss = self(inputs, targets)
                losses[k] = loss.item()
            out[stage] = losses.mean()

        self.train(mode)
        return out

    @torch.no_grad()
    def generate(self, max_tokens, ctx=None, temp=1.0):
        self.eval()

        if ctx is None:
            ctx = torch.zeros(
                (1, 1),
                dtype=torch.long,
                device=self.params.device)

        for _ in range(max_tokens):
            inputs = ctx[:, -self.params.ctx_length:] # crop ctx to the last ctx_length tokens
            logits, _ = self(inputs) # get the predictions
            logits = logits[:, -1, :] # get the last time steps
            logits /= temp # apply temperature (>1 more random, <1 less random)
            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities
            preds = torch.multinomial(probs, num_samples=1) # sample predictions from the distribution
            ctx = torch.cat((ctx, preds), dim=1) # append predictions to the running sequence

        return self.dataset.decode(ctx[0].tolist())
    
    def complete(self, prompt, max_tokens, temp=1.0):
        ctx = None
        if prompt != '':
            ctx = torch.tensor(
                self.dataset.encode(prompt),
                dtype=torch.long,
                device=self.params.device).unsqueeze(0)

        output = self.generate(max_tokens, ctx, temp)
        return output[len(prompt):]
