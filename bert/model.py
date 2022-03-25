import sys
sys.path.insert(0, "/Users/huaxuanwang/Project/NLP/")
import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention
from feedforward import PositionwiseFeedForward
from embedding import Bert_Embedding
from utils import attention_mask


class BertEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_head=8, dropout_rate=0.1):
        super().__init__()
        
        self.att = MultiHeadAttention(embedding_dim, num_head)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc = PositionwiseFeedForward(embedding_dim, hidden_dim, dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, x, x_mask=None):
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


class Bert(nn.Module):
    def __init__(self, vocab_size, max_len, num_layer=5, embedding_dim=512, hidden_dim=2048, num_head=8, dropout_rate=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx=pad_idx

        self.emb = Bert_Embedding(vocab_size=vocab_size,
                                  max_len=max_len,
                                  embedding_dim=embedding_dim)
        
        self.layers = nn.ModuleList([BertEncoderBlock(embedding_dim=embedding_dim,
                                                      hidden_dim=hidden_dim,
                                                      num_head=num_head,
                                                      dropout_rate=dropout_rate)
                                    for _ in range(num_layer)])

        self.output_linear = nn.Linear(in_features=embedding_dim,
                                       out_features=embedding_dim)
        self.activation = nn.Tanh()
    def forward(self, x, seg):
        inputs = x
        x = self.emb(x, seg)
        
        mask = attention_mask(inputs, inputs, self.pad_idx)

        for layer in self.layers:
            x = layer(x, mask)
        # 取出<cls>的embedding, 并通过一层fc和tanh. 至于为什么用tanh，原作者：https://github.com/google-research/bert/issues/43
        # x:(batch_size, seq_len, emb_dim)
        # pooled_h:(batch_size, emb_dim)
        pooled_h = x[:, 0]

        output = self.output_linear(pooled_h)
        output = self.activation(output)

        #output: (batch_size, emb_dim)
        return output
if __name__ == '__main__':

    x = torch.tensor([[1, 2, 3, 4, 5, 6, 3, 0], [12, 13, 14, 15, 16, 17, 0, 0]], dtype=torch.long)
    seg = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)

    emb = Bert(50, 100)

    y = emb(x, seg)

    print(y.shape)

