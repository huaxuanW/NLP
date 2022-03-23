import torch
import torch.nn as nn


class Token_Embedding(nn.Embedding):

    def __init__(self, vocab_size, embedding_dim):

        super(Token_Embedding, self).__init__(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=1
        )
        # return: (batch_size, seq_len, embedding_dim)


if __name__ == "__main__":

    sentence = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    token_emb = Token_Embedding(20, 512)
    res = token_emb(sentence)

    print(res.shape)
