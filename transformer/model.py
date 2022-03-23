import torch
import torch.nn as nn

from feedforward import PositionwiseFeedForward
from layer_norm import LayerNorm
from multihead_attention import MultiHeadAttention
from transfomer_embedding import Transfomer_Embedding


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_head, dropout_rate):
        super().__init__()

        self.att = MultiHeadAttention(embedding_dim, num_head)
        self.norm1 = LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc = PositionwiseFeedForward(embedding_dim, hidden_dim, dropout_rate)
        self.norm2 = LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x,  x_mask):
        # 1. multi-head attention
        x_copy = x
        x = self.att(q=x, k=x, v=x, mask=x_mask)

        # 2. add and norm
        x = self.norm1(x + x_copy)
        x = self.dropout1(x)

        # 3. pointwise feed forward
        x_copy = x
        x = self.fc(x)
        # 4. add and norm
        x = self.norm2(x + x_copy)
        x = self.dropout2(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_head, dropout_rate):
        super().__init__()

        self.att1 = MultiHeadAttention(embedding_dim, num_head)
        self.norm1 = LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.att2 = MultiHeadAttention(embedding_dim, num_head)
        self.norm2 = LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc = PositionwiseFeedForward(embedding_dim, hidden_dim, dropout_rate)
        self.norm3 = LayerNorm(embedding_dim)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, trg, src, mask_src, mask_trg):
        # 1. multi-head attention using only target sentence
        x_copy = trg
        x = self.att1(q=trg, k=trg, v=trg, mask=mask_trg)

        # 2. add and norm
        x = self.norm1(x + x_copy)
        x = self.dropout1(x)

        # 3. multi-head attention between target and source sentences
        x_copy = x
        x = self.att2(q=x, k=src, v=src, mask=mask_src)

        # 4. add and norm
        x = self.norm2(x + x_copy)
        x = self.dropout2(x)

        # 5. positionwise feed forward
        x_copy = x
        x = self.fc(x)
        # 6. add and norm
        x = self.norm3(x + x_copy)
        x = self.dropout3(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, num_layers, vocab_size, embedding_dim, max_len, hidden_dim, num_head, dropout_rate, device):
        super().__init__()

        self.emb = Transfomer_Embedding(vocab_size=vocab_size, 
                                        embedding_dim=embedding_dim,
                                        max_len=max_len,
                                        dropout_rate=dropout_rate,
                                        device=device)

        self.layers = nn.ModuleList([EncoderBlock(embedding_dim=embedding_dim, 
                                                  hidden_dim=hidden_dim,
                                                  num_head=num_head, 
                                                  dropout_rate=dropout_rate)
                                    for _ in range(num_layers)])


    def forward(self, x, mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, num_layers, vocab_size, embedding_dim, max_len, hidden_dim, num_head, dropout_rate, device):
        super().__init__()

        self.emb = Transfomer_Embedding(vocab_size=vocab_size, 
                                        embedding_dim=embedding_dim,
                                        max_len=max_len,
                                        dropout_rate=dropout_rate,
                                        device=device)

        self.layers = nn.ModuleList([DecoderBlock(embedding_dim=embedding_dim, 
                                                  hidden_dim=hidden_dim,
                                                  num_head=num_head, 
                                                  dropout_rate=dropout_rate)
                                    for _ in range(num_layers)])

        self.output_layer = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, trg, src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
        
        return self.output_layer(trg)

class Transformers(nn.Module):
    def __init__(self):
        super().__init__()