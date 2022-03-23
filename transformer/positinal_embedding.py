import torch
import torch.nn as nn


class Positional_Embedding(nn.Module):
    def __init__(self, embedding_dim, max_len, device):
        super().__init__()

        # 与输入矩阵大小相同（用于与输入矩阵相加）
        self.encoding = torch.zeros(
            (max_len, embedding_dim), device=device, requires_grad=False)

        # 1D=>2D unsqueze表示单词的位置
        pos = torch.arange(0, max_len, device=device,
                           dtype=torch.float32).unsqueeze(1)

        # “i”是指embedding_dim的索引（例如embedding size=50，i=[0,50]）
        # “step=2”表示“i”乘以2（与2*i相同）
        _2i = torch.arange(0, embedding_dim, 2,
                           device=device, dtype=torch.float32)

        # 计算位置编码以考虑单词的位置信息
        self.encoding[:, 0::2] = torch.sin(
            pos / (10000 ** (_2i / embedding_dim)))
        self.encoding[:, 1::2] = torch.cos(
            pos / (10000 ** (_2i / embedding_dim)))

    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        # self.encoding: (max_len, embedding_dim)
        return self.encoding[:seq_len, :]
        # return: (seq_len, embedding_dim), 因此可以添加到 (batch_size, seq_len, embedding_dim)


if __name__ == "__main__":

    embedding = torch.rand((10, 50))
    positional_emb = Positional_Embedding(
        embedding_dim=512, max_len=100, device="cpu")
    res = positional_emb(embedding)

    print(res.shape)
