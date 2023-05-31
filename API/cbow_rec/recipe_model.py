import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, window_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.window_size = window_size
        
    def forward(self, x):
        x = self.embedding(x)
        x = torch.sum(x, dim=1)
        x = self.linear(x)
        log_probs = F.log_softmax(x, dim=1)
        return log_probs
    