import torch.nn as nn
import torch


class Bert_Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, embedding_dim):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.positional_embedding = nn.Embedding(max_len, embedding_dim)
        self.segment_embedding = nn.Embedding(2, embedding_dim)

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, seg):
        # seg: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # generate position index, (batch_size, seq_len)
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        pos = pos.repeat(batch_size, 1)
        
        # get embedding
        # token_emb: (batch_size, seq_len, embed_dim)
        tok_emb = self.token_embedding(x)
        # pos_emb: (batch_size, seq_len, embed_dim)
        pos_emb = self.positional_embedding(pos)
        # seg_emb: (batch_size, seq_len, embed_dim)
        seg_emb = self.segment_embedding(seg)
        
        emb = tok_emb + pos_emb + seg_emb

        return self.norm(emb)

if __name__ == '__main__':
    
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [12, 13, 14, 15, 16, 17, 18, 19]], dtype=torch.long)
    seg = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)

    emb = Bert_Embedding(50, 100, 512)

    y = emb(x, seg)

    print(y.shape) #(batch_size:2, seq_len:8, emb_dim:512)