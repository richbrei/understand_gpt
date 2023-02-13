from dataclasses import dataclass
from turtle import forward
import numpy as np  

import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        logits = self.token_embedding_table(inputs) # (Batch, Context, Vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, inputs, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(inputs)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            inputs = torch.cat([inputs,next_token], dim=1)

        return inputs

@dataclass
class GPTConfig:
    context_length: int = 8
    vocab_size: int = 65
    head_size: int = 12
    num_heads: int = 4
    n_embd: int = 32

class SelfAttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, context_length) -> None:
        super().__init__()
        self.key   = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_length,context_length)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (batch, context, head)
        q = self.query(x) # (batch, context, head)
        v = self.value(x) # (batch, context, head)

        attention_filter = q @ k.transpose(-2,-1) * C**-.5 # (batch, context, context)
        attention_filter = attention_filter.masked_fill(self.tril[:T,:T]==0, float("-inf")) # (batch, context, context)
        attention_filter = F.softmax(attention_filter, dim=-1) # (batch, context, context)
        out = attention_filter @ v # (batch, context, head)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, head_size, context_length, num_heads) -> None:
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(n_embd, head_size, context_length) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.context_length, config.n_embd)
        self.sa_heads = MultiHeadAttention(config.n_embd, config.head_size, config.context_length, config.num_heads)
        self.lm_head = nn.Linear(config.head_size*config.num_heads, config.vocab_size)
        self.config = config

    def forward(self, inputs, targets=None):
        B, T = inputs.shape
        tok_embd = self.token_embedding_table(inputs) # (batch, context, n_embd)
        pos_embd = self.position_embedding_table(torch.arange(T)) # (context, n_embd)
        x = tok_embd + pos_embd # (batch, context, n_embd)
        x = self.sa_heads(x) # (batch, context, n_embd)
        logits = self.lm_head(x) # (batch, context, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, inputs, max_new_tokens):
        for _ in range(max_new_tokens):
            cropped_inputs = inputs[:,-self.config.context_length:]
            logits, loss = self(cropped_inputs)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            inputs = torch.cat([inputs,next_token], dim=1)

        return inputs