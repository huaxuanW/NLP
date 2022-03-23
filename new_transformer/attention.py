import math

import torch
import torch.nn.functional as F


def DotProductAttention(q, k, v, mask=None):

    batch_size, num_head, seq_len, embedding_dim = k.size()

    # 使用 query 和 key^T 计算相似度
    k_t = k.transpose(2, 3)
    score = (q @ k_t) / math.sqrt(embedding_dim)
    # print(f"score shape: {score.shape}")

    # 应用masking（可选）
    if mask is not None:
        score = score.masked_fill(mask == 0, 1e-9)

    # 使用softmax计算概率分布
    score = F.softmax(score, dim=-1)

    # 将结果与 value 相乘
    v = score @ v

    return v


if __name__ == "__main__":

    batch_size, num_head, seq_len, embedding_dim = 2, 2, 5, 10
    q = torch.rand((batch_size, num_head, seq_len, embedding_dim))
    k = torch.rand((batch_size, num_head, seq_len - 1, embedding_dim))
    v = torch.rand((batch_size, num_head, seq_len - 1, embedding_dim))
    mask = torch.zeros((batch_size, num_head, 5, 4),
                       dtype=torch.uint8)  # or dtype=torch.ByteTensor
    mask[:, :, 0, 0] = 1
    mask[:, :, 1, 1] = 1
    mask[:, :, 3, 2] = 1

    print(f"mask shape: {mask.shape}")
    res = DotProductAttention(q, k, v, mask.T)

    print(res.shape)
