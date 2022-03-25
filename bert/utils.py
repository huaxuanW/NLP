import torch
import math
import torch.nn as nn

def attention_mask(seq_q, seq_k, pad_idx=0):
    batch_size, len_q = seq_q.size()
    len_k = seq_k.size(1)

    # (batch_size, 1, 1, len_k)
    seq_k = seq_k.ne(pad_idx).reshape(batch_size, 1, 1, len_k)
    # (batch_size, 1, len_q, len_k)
    mask = seq_k.repeat(1, 1, len_q, 1)
    
    # # (batch_size, 1, len_q, 1)
    # seq_q = seq_q.ne(pad_idx).reshape(batch_size, 1, len_q, 1)
    # # (batch_size, 1, len_q, len_k)
    # seq_q = seq_q.repeat(1, 1, 1, len_k)
    
    # mask = seq_k & seq_q
    return mask


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LayerNorm(nn.Module):

    def __init__(self, embedding_dim, eps=1e-9):
        super().__init__()

        self.beta = nn.Parameter(torch.ones(embedding_dim))
        self.gamma = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.beta * (x - mean) / (std + self.eps) + self.gamma


if __name__ == "__main__":
    seq_q = torch.tensor([[1, 2, 3, 4, 5]])
    seq_k = torch.tensor([[0, 1, 2, 3, 4]])

    mask = attention_mask(seq_q, seq_k)

    print(mask.shape)

    print(mask)