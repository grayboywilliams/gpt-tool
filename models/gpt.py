import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from .params import *
from .dataset import *
from components.block import *
from models.logger import *
from constants.constants import *

class GPTLanguageModel(nn.Module):

    def __init__(self, logger, params: Hyperparameters, dataset: Dataset):
        super().__init__()
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.logger = logger
        self.params = params
        self.dataset = dataset

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.dataset.vocab_size, self.params.num_dim)
        self.position_embedding_table = nn.Embedding(params.ctx_length, self.params.num_dim)
        self.blocks = nn.Sequential(*[Block(self.params) for _ in range(self.params.num_layer)])
        self.ln_f = nn.LayerNorm(self.params.num_dim) # final layer norm
        self.lm_head = nn.Linear(self.params.num_dim, self.dataset.vocab_size)
        self.apply(self._init_weights)
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.params.learning_rate)
        self.to(self.params.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.params.device)) # (T,C)
        x = token_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # does softmax multinomial internally

        return logits, loss
    
    def begin_train(self):
        start_time = time.time()
        self.logger.log(SUMMARY, f"num_batch: {self.params.num_batch}, " +
                        f"eval_interval: {self.params.eval_interval}, " +
                        f"eval_iters: {self.params.eval_iters}")

        for iter in range(self.params.num_batch):

            # every once in a while evaluate the loss on train and val sets
            if iter % self.params.eval_interval == 0 or iter == self.params.num_batch - 1:
                losses = self.estimate_loss()
                elapsed_time = time.time() - start_time
                elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                self.logger.log(SUMMARY, f"step {iter}: train loss {losses[train]:.4f}, " +
                           f"val loss {losses[val]:.4f}, time {elapsed_time_str}")

            # sample a batch of data
            xb, yb = self.dataset.get_batch('train')

            # evaluate the loss
            _, loss = self(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        
        losses = self.estimate_loss(True)
        self.logger.log(SUMMARY, f"final: test loss {losses[test]:.4f}")

    @torch.no_grad()
    def estimate_loss(self, test=False):
        out = {}
        stages = [train, val] if test == False else [test]
        for stage in stages:
            losses = torch.zeros(self.params.eval_iters)
            for k in range(self.params.eval_iters):
                X, Y = self.dataset.get_batch(stage)
                _, loss = self(X, Y)
                losses[k] = loss.item()
            out[stage] = losses.mean()
        self.train(False)
        return out

    def generate(self, tokens, ctx=None, temp=1.0):
        if ctx is None:
            ctx = torch.zeros(
                (1, 1),
                dtype=torch.long,
                device=self.params.device)

        # ctx is (B, T) array of indices in the current context
        for _ in range(tokens):
            # crop ctx to the last ctx_length tokens
            idx_cond = ctx[:, -self.params.ctx_length:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply temperature (>1 for more randomness, <1 for less randomness)
            logits /= temp
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            ctx = torch.cat((ctx, idx_next), dim=1) # (B, T+1)

        return self.dataset.decode(ctx[0].tolist())
    
    def complete(self, prompt, tokens, temp=1.0):
        ctx = torch.tensor(
            self.dataset.encode(prompt),
            dtype=torch.long,
            device=self.params.device).unsqueeze(0)

        output = self.generate(tokens, ctx, temp)
        return output[len(prompt):]

    def save_parameters(self, name=None):
        if name != None:
            self.params.name = name

        filename = os.path.join(self.script_dir, '../checkpoints', name, 'checkpoint.pth')

        torch.save(self.state_dict(), filename) # save weights
        self.params.save_config() # save config
        save_summary_log(self.params.name) # save logs

        return 'Model parameters saved successfully.'

    def load_parameters(self, name):
        try:
            filename = os.path.join(self.script_dir, '../checkpoints', name, 'checkpoint.pth')
            self.load_state_dict(torch.load(filename))
            self.eval()  # Set the model to evaluation mode after loading
            return 'Model parameters loaded successfully.'
        except FileNotFoundError:
            return 'Model parameters file not found.'
