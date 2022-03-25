import torch
import torch.nn as nn

from transformer.positinal_embedding import Positional_Embedding
from transformer.token_embeddig import Token_Embedding


class Transfomer_Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len, dropout_rate, device):
        super().__init__()

        self.token_emb = Token_Embedding(vocab_size, embedding_dim)
        self.pos_emb = Positional_Embedding(embedding_dim, max_len, device)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(x)

        return self.dropout(token_emb + pos_emb)

if __name__== "__main__":

    sentence = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    emb = Transfomer_Embedding(20, 512, 10, 0.1, "cpu")
    res = emb(sentence)

    print(res.shape)