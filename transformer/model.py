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
        x = self.dropout1(x)
        x = self.norm1(x + x_copy)

        # 3. pointwise feed forward
        x_copy = x
        x = self.fc(x)
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + x_copy)

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

    def forward(self, trg, src, mask_trg, mask_src):
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
            # print(x.shape, mask.shape)
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
    def __init__(self, pad_idx, src_vocab_size, trg_vocab_size, num_layers, 
                 embedding_dim, max_len, hidden_dim, num_head, dropout_rate, device):
        super().__init__()
    
        self.encoder = EncoderLayer(num_layers=num_layers,
                                    vocab_size=src_vocab_size,
                                    embedding_dim=embedding_dim,
                                    max_len=max_len,
                                    hidden_dim=hidden_dim,
                                    num_head=num_head,
                                    dropout_rate=dropout_rate,
                                    device=device)

        self.decoder = DecoderLayer(num_layers=num_layers,
                                    vocab_size=trg_vocab_size,
                                    embedding_dim=embedding_dim,
                                    max_len=max_len,
                                    hidden_dim=hidden_dim,
                                    num_head=num_head,
                                    dropout_rate=dropout_rate,
                                    device=device)
        
        self.pad_idx = pad_idx
        self.device = device

    def forward(self, src, trg):
        src_mask = self.generate_mask(src, src)

        src_trg_mask = self.generate_mask(trg, src)

        trg_mask = self.generate_mask(trg, trg) * self.make_no_peak_mask(trg, trg)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)

        return output
    
    def generate_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        
        # to shape(batch_size, 1, 1, len_k)
        k = k.ne(self.pad_idx).reshape(-1, 1, 1, len_k)
        # to shape(batch_size, 1, len_q, len_k)
        k = k.repeat(1, 1, len_q, 1)

        # to shape(batch_size, 1, len_q, 1)
        q = q.ne(self.pad_idx).reshape(-1, 1, len_q, 1)
        # to shape(batch_size, 1, len_q, len_k)
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask


if __name__== "__main__":
    pad_idx=0
    src_vocab_size=20
    trg_vocab_size=20
    num_layers=1
    embedding_dim=512
    max_len=60
    hidden_dim=2048
    num_head=8
    dropout_rate=0.1
    device="cpu"

    src = torch.tensor([[1, 3, 5, 7, 0, 0, 1, 1, 2], [2, 4, 6, 8, 10, 0, 1, 1, 1]])
    trg = torch.tensor([[2, 4, 6, 8, 10, 1, 0], [3, 3, 1, 6, 9, 5, 0]])
    model = Transformers(pad_idx,
                        src_vocab_size,
                        trg_vocab_size,
                        num_layers,
                        embedding_dim,
                        max_len,
                        hidden_dim,
                        num_head,
                        dropout_rate,
                        device)
    
    res = model(src, trg)

    print(res.shape)
