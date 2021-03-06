# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mMeTWAtvkoAl8C7BvOAFucIj89JS1cPJ

# package
"""
# importing sys
import sys
  
# adding Folder_2 to the system path
sys.path.insert(0, '/Users/huaxuanwang/Project/NLP/')

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import math
import time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

"""# load model"""

# !cp /content/drive/MyDrive/深度学习/NLP/transformers/attention.py .
# !cp /content/drive/MyDrive/深度学习/NLP/transformers/feedforward.py .
# !cp /content/drive/MyDrive/深度学习/NLP/transformers/layer_norm.py .
# !cp /content/drive/MyDrive/深度学习/NLP/transformers/model.py .
# !cp /content/drive/MyDrive/深度学习/NLP/transformers/multihead_attention.py .
# !cp /content/drive/MyDrive/深度学习/NLP/transformers/positinal_embedding.py .
# !cp /content/drive/MyDrive/深度学习/NLP/transformers/token_embeddig.py .
# !cp /content/drive/MyDrive/深度学习/NLP/transformers/transfomer_embedding.py .

from transformer.model import Transformers

"""#data preparation

## transformation
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install -U spacy
# !python -m spacy download en_core_web_sm
# !python -m spacy download de_core_news_sm

tokenizers = (get_tokenizer('spacy', language='en_core_web_sm'), 
              get_tokenizer('spacy', language='de_core_news_sm'))

train_list = [pair for pair in Multi30k(split='train', language_pair=('en', 'de'))]

valid_list = [pair for pair in Multi30k(split='train', language_pair=('en', 'de'))]

def build_vocab(data, specials, tokenizers):
  vocab_src = build_vocab_from_iterator(data_iter(0, data, tokenizers), specials=specials)
  vocab_trg = build_vocab_from_iterator(data_iter(1, data, tokenizers), specials=specials)
  
  vocab_src.set_default_index(0)
  vocab_trg.set_default_index(0)

  return vocab_src, vocab_trg

def data_iter(idx, data, tokenizers):
  for pair in data:
    yield tokenizers[idx](pair[idx])

def tokenize_data(data, vocab_src, vocab_trg, tokenizers):
  res = []
  for pair in data:
    src_tensor = torch.tensor([vocab_src[token] for token in tokenizers[0](pair[0].rstrip("\n"))], dtype=torch.long)
    trg_tensor = torch.tensor([vocab_trg[token] for token in tokenizers[1](pair[1].rstrip("\n"))], dtype=torch.long)
    res.append((src_tensor, trg_tensor))
  return res

specials=['<unk>', '<pad>', '<bos>', '<eos>']

vocab_en, vocab_de = build_vocab(train_list, specials, tokenizers)

train_set = tokenize_data(train_list, vocab_en, vocab_de, tokenizers)
valid_set = tokenize_data(valid_list, vocab_en, vocab_de, tokenizers)

"""## hyper parameter"""

PAD_IDX = vocab_en['<pad>']
BOS_IDX = vocab_en['<bos>']
EOS_IDX = vocab_en['<eos>']
BATCH_SIZE = 128

"""## trainer"""

def generate_batch(data_batch):
  batch_src, batch_trg = [], []
  for (src, trg) in data_batch:
    batch_src.append(torch.cat([torch.tensor([BOS_IDX]), src, torch.tensor([EOS_IDX])], dim=0))
    batch_trg.append(torch.cat([torch.tensor([BOS_IDX]), trg, torch.tensor([EOS_IDX])], dim=0))
  batch_src = pad_sequence(batch_src, padding_value=PAD_IDX)
  batch_trg = pad_sequence(batch_trg, padding_value=PAD_IDX)
  return batch_src, batch_trg

def train(model, dataloader, optimizer,
          criterion, clip, device):
  
  model.train()

  epoch_loss = 0

  dataloader = tqdm(dataloader)

  for src, trg in dataloader:
      
    src, trg = src.T.to(device), trg.T.to(device)

    optimizer.zero_grad()

    # print(f"src shape: {src.shape}")
    # print(f"trg shape: {trg.shape}")
    # print("-"*30)

    output = model(src, trg)

    output = output[1:].reshape(-1, output.shape[-1])
    trg = trg[1:].reshape(-1)

    loss = criterion(output, trg)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()

    epoch_loss += loss.item()

  return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):

  model.eval()

  epoch_loss = 0
  dataloader = tqdm(dataloader)
  with torch.no_grad():

      for src, trg in dataloader:
          src, trg = src.T.to(device), trg.T.to(device)

          output = model(src, trg) #turn off teacher forcing

          output = output[1:].reshape(-1, output.shape[-1])
          trg = trg[1:].reshape(-1)

          loss = criterion(output, trg)

          epoch_loss += loss.item()

  return epoch_loss / len(dataloader)

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

"""# Train"""

pad_idx = vocab_en['<pad>']
src_vocab_size = len(vocab_en)
trg_vocab_size = len(vocab_de)
num_layers = 6
embedding_dim = 512
max_len = 256
hidden_dim = 512
num_head = 8
dropout_rate = 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoches = 3
clip = 1.0
weight_decay = 5e-4
inf = float('inf')

mod = Transformers(pad_idx,
                   src_vocab_size,
                   trg_vocab_size,
                   num_layers,
                   embedding_dim,
                   max_len,
                   hidden_dim,
                   num_head,
                   dropout_rate,
                   device)

print(f'The model has {count_parameters(mod):,} trainable parameters')
mod.apply(init_weights)
mod = mod.to(device)

optimizer = optim.Adam(params=mod.parameters(),
                      lr=init_lr,
                      weight_decay=weight_decay,
                      eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

train_iter = DataLoader(train_set, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(valid_set, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)

best_valid_loss = float('inf')

for epoch in range(epoches):

    start_time = time.time()

    train_loss = train(mod, train_iter, optimizer, criterion, clip, device)
    valid_loss = evaluate(mod, valid_iter, criterion, device)

    end_time = time.time()
    if epoch > warmup:
      scheduler.step(valid_loss)
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

