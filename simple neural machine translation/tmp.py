from unicodedata import bidirectional
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

F.cross_entropy()
t = torch.arange(0, 24).reshape(2, 3, 4) #(l, b, vocab)
t
t.sum(0).shape
index = torch.tensor([
    [0, 2, 1],
    [3, 1, 2]
])
prob = torch.gather(t, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
prob
prob.sum(0)

x = torch.rand((10, 512))
enc_hidden_proj = torch.rand(10, 50, 100)

lstm = nn.LSTMCell(
    input_size= 512,
    hidden_size= 100,
    bias=False
)

(last_hidden, last_cell) = lstm(x)

score = torch.bmm(last_hidden.unsqueeze(1), enc_hidden_proj.permute(0, 2, 1)).squeeze(1)

print(enc_hiddens.shape)
print(last_hidden.shape)
print(last_cell.shape)

init_decoder_hidden = torch.cat([last_hidden[0], last_hidden[1]], dim=-1)
print(init_decoder_hidden.shape)