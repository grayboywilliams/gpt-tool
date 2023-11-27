import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from .params import *
from .dataset import *
from components.block import *

class GPTLanguageModel(nn.Module):

    def __init__(self, params: Hyperparameters, dataset: Dataset):
        super().__init__()
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.params = params
        self.dataset = dataset

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.dataset.vocab_size, self.params.num_dim)
        self.position_embedding_table = nn.Embedding(params.block_size, self.params.num_dim)
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
        for iter in range(self.params.num_batch):

            # every once in a while evaluate the loss on train and test sets
            if iter % self.params.eval_interval == 0:
                losses = self.estimate_loss()
                elapsed_time = time.time() - start_time
                elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}, time {elapsed_time_str}")

            # sample a batch of data
            xb, yb = self.dataset.get_batch('train')

            # evaluate the loss
            _, loss = self(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        for stage in ['train', 'test']:
            losses = torch.zeros(self.params.eval_iters)
            for k in range(self.params.eval_iters):
                X, Y = self.dataset.get_batch(stage)
                _, loss = self(X, Y)
                losses[k] = loss.item()
            out[stage] = losses.mean()
        self.train(False)
        return out

    def generate(self, tokens, ctx=None):
        if ctx is None:
            ctx = torch.zeros(
                (1, 1),
                dtype=torch.long,
                device=self.params.device)

        # ctx is (B, T) array of indices in the current context
        for _ in range(tokens):
            # crop ctx to the last block_size tokens
            idx_cond = ctx[:, -self.params.block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            ctx = torch.cat((ctx, idx_next), dim=1) # (B, T+1)

        return self.dataset.decode(ctx[0].tolist())
    
    def complete(self, prompt, tokens):
        ctx = torch.tensor(
            self.dataset.encode(prompt),
            dtype=torch.long,
            device=self.params.device).unsqueeze(0)

        output = self.generate(tokens, ctx)
        return output[len(prompt):]

    def save_parameters(self, name=None):
        if name != None:
            self.params.checkpoint_name = name

        filename = os.path.join(self.script_dir, '../checkpoints', self.params.checkpoint_name)
        torch.save(self.state_dict(), filename)
        return 'Model parameters saved successfully.'

    def load_parameters(self, name=None):
        try:
            if name != None:
                self.params.checkpoint_name = name

            filename = os.path.join(self.script_dir, '../checkpoints', self.params.checkpoint_name)
            self.load_state_dict(torch.load(filename))
            self.eval()  # Set the model to evaluation mode after loading
            return 'Model parameters loaded successfully.'
        except FileNotFoundError:
            return 'Model parameters file not found.'