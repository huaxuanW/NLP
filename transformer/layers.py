import torch

import torch.nn as nn

def init_weight(layer):
    """
    it is necessary to init weight using some distribution, 
    since the default initialized weights from `torch.tensor` can be in large range,
    which may cause the model hard to converge.
    """
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

def attention(q, k, v, dk, mask, dropout):
    # scale inplace
    q.mul_(dk ** -0.5)
    
    # calculate similarity score and softmax
    score = torch.matmul(q, k.transpose(-2, -1)) # q:(batch_size, num_head, seqlen, dk) k.T: (batch_size, num_head, dk, seqlen) -> score:(batch_size, num_head, seqlen, seqlen)
    score = torch.softmax(score, dim=-1) # calculate probability on the last dimension

    # apply mask inplace
    if mask is not None:
        score.masked_fill_(mask.unsqueeze(1), -1e9)

    # apply dropout
    if dropout is not None:
        scores = dropout(scores)

    # apply similarity score on value
    result = torch.matmul(score, v) # score:(batch_size, num_head, seqlen, seqlen), v:(batch_size, num_head, seqlen, dk) -> result:(batch_size, num_head, seqlen, dk)
    
    return result

class FeedForward(nn.Module):
    """
    implement point-wise Feed Forward sub-layer in transformer

    point-wise == apply same fc for each word

    map the extracted feature into the desired semantic space

    """
    def __init__(self, inputs_dim=512, outputs_dim=512, hidden_dim=2048, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(inputs_dim, hidden_dim,)

        self.activation = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, outputs_dim,)

        self.dropout = nn.Dropout(dropout)

        init_weight(self.fc1)
        init_weight(self.fc2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiheadAttention(nn.Module):
    """
    implement Multi-head attention sub-layer in transformer

    purpose: extract sequential feature that is needed from input sequence

    speed: comparing to RNN that sequentially pass the feature, attention globally extract feature 

    performance: like multi-channel CNN, multi-head attention extract different feature pattern 
    """
    def __init__(self, inputs_dim, dropout, num_head=8):
        super().__init__()

        self.dk = inputs_dim // num_head
        self.num_head = num_head
        
        self.query = nn.Linear(inputs_dim, self.dk, bias=False)
        self.key = nn.Linear(inputs_dim, self.dk, bias=False)
        self.value = nn.Linear(inputs_dim, self.dk, bias=False)

        init_weight(self.query)
        init_weight(self.key)
        init_weight(self.value)

        self.dropout = nn.Dropout(dropout)

        self.output_linear = nn.Linear(self.dk * num_head, inputs_dim, bias=False)

        init_weight(self.output_linear)

    def forward(self, q, k, v, mask):

        batch_size = q.size(0)
        
        # perform linear operation and split into N heads
        q = self.query(q.view(batch_size, -1, self.num_head, self.dk))
        k = self.query(k.view(batch_size, -1, self.num_head, self.dk))
        v = self.query(v.view(batch_size, -1, self.num_head, self.dk))

        # transpose from (batch * seqlen * num_head * dk) -> (batch * num_head * seqlen * dk)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        #calculate attention
        score = attention(q, k, v, self.dk, mask, self.dropout)

        return




batch, num_head, sentence_length, embedding_dim = 1, 8, 5, 3
q = v = k = torch.randn(batch, num_head, sentence_length, embedding_dim)


attention(q,v,k,None,3,None)