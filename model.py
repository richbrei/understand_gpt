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

class GPT(nn.Module):

    def __init__(self):
        pass