import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

        self.eps = 1e-9

    def forward(self, x):
        # -1 -> 在最后一个维度上
        x_mean = x.mean(-1, keepdim=True)
        x_std = x.std(-1, keepdim=True)

        output = (x - x_mean) / (x_std + self.eps)
        return self.gamma * output + self.beta


if __name__ == "__main__":

    embedding = torch.rand((3, 10, 512))
    norm = LayerNorm(embedding_dim=512)
    res = norm(embedding)

    print(res.shape)
