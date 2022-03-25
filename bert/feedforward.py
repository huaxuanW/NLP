import torch.nn as nn
from utils import gelu

class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)

        self.gelu = gelu
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


if __name__ == "__main__":
    import torch

    embedding = torch.rand((10, 50, 512))
    fc = PositionwiseFeedForward(
        embedding_dim=512, hidden_dim=2048, dropout_rate=0.1)
    res = fc(embedding)

    print(res.shape)
