import torch.nn as nn
import math
import torch.nn.functional as F


def DotProductAttention(q, k, v, mask=None):

    batch_size, num_head, seq_len, embedding_dim = k.size()

    # 使用 query 和 key^T 计算相似度
    k_t = k.transpose(2, 3)
    score = (q @ k_t) / math.sqrt(embedding_dim)
    # print(f"q shape: {q.shape}")
    # print(f"k_t shape: {k_t.shape}")
    # print(f"score shape: {score.shape}")
    # print(f"mask shape: {mask.shape}")
    # print(f"-"*18)
    
    # 应用masking（可选）
    if mask is not None:
        score = score.masked_fill(mask == 0, 1e-9)

    # 使用softmax计算概率分布
    score = F.softmax(score, dim=-1)

    # 将结果与 value 相乘
    v = score @ v

    return v


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_head):
        super().__init__()

        self.new_emb_dim = embedding_dim // num_head

        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)

        self.output = nn.Linear(self.new_emb_dim * num_head, embedding_dim)

        self.num_head = num_head

    def forward(self, q, k, v, mask=None):

        # 1.qkv与矩阵相乘
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2.将qkv分成n_head
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3.计算相似度
        att_output = DotProductAttention(q, k, v, mask)

        # 4.将结果并在一起
        att_output = self.concat(att_output)

        # 5.通过最后linear层
        return self.output(att_output)

    def split(self, matrix):

        batch_size, seq_len, embedding_dim = matrix.size()

        matrix = matrix.reshape(
            batch_size, seq_len, self.num_head, self.new_emb_dim).transpose(1, 2)

        return matrix

    def concat(self, matrix):

        batch_size, num_head, seq_len, new_emb_dim = matrix.size()

        matrix = matrix.transpose(1, 2).contiguous().reshape(
            batch_size, seq_len, new_emb_dim * num_head)

        return matrix