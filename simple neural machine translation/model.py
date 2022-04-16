from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn as nn
import torch
import torch.nn.functional as F

class ModelEmbedding(nn.Module):
    """ Embedding Layer that convert inputs words to their embeddings
    """
    def __init__(self, embed_size, vocab):
        """ Initialize the Embedding Layer
        @param embed_size (int): embedding size
        @param vocab (Vocab): Vocabulary containing source and target language
        """
        super().__init__()
        self.source = nn.Embedding(
            num_embeddings=len(vocab.src),
            embedding_dim=embed_size,
            padding_idx=vocab.src['<pad>']
        )
        self.target = nn.Embedding(
            num_embeddings=len(vocab.tgt),
            embedding_dim=embed_size,
            padding_idx=vocab.tgt['<pad>']
        )


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidirectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model
    """
    def __init__(self, embed_size, vocab, hidden_size, dropout_rate):
        """ Initialize NMT Model
        @param embed_size (int): embedding size
        @param vocab (Vocab): Vocabulary containing source and target language
        @param hidden_size (int): the size of hidden state
        @param dropout_rate (float): dropout probability
        """
        super().__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        # embedding layer
        self.model_embeddings = ModelEmbedding(embed_size, vocab)
        # bidirectional lstm layer
        self.encoder = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size,
            bias=True,
            bidirectional=True
        )
        # unidirectional lstm layer
        self.decoder = nn.LSTMCell(
            input_size=embed_size + hidden_size,
            hidden_size=hidden_size,
            bias=True,
        )
        self.h_linear = nn.Linear(
            in_features=hidden_size*2, #bidirection
            out_features=hidden_size,
            bias=False 
        )
        self.c_linear = nn.Linear(
            in_features=hidden_size*2,
            out_features=hidden_size,
            bias=False,
        )
        self.att_linear = nn.Linear(
            in_features=hidden_size*2,
            out_features=hidden_size,
            bias=False,
        )
        self.combined_output_linear =nn.Linear(
            in_features=hidden_size*3,
            out_features=hidden_size,
            bias=False
        )
        self.final_output = nn.Linear(
            in_features=hidden_size,
            out_features=len(vocab.tgt),
            bias=False
        )
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, source, target):
        """ Take a mini-batch of source and target sentence pairs,
            compute the log-likelihood of target sentence under the
            language models learned by the NMT model.
            @param source (List[List[str]]): list of list of source tokens
            @param target (List[List[str]]): list of list of target tokens
            @return scores (Tensor): a tensor of shape(batch size, ) representing the log-likelihood of generating the gold-standard tareget sentence for each example in the input batch.
        """
        source_lengths = [len(s) for s in source]

        source_padded = self.vocab.src.to_input_tensor(source, device=self.device) 
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)

        X = self.model_embeddings.source(source_padded)

        enc_hiddens, (last_hidden, last_cell) = self.encoder(pack_padded_sequence(X, lengths=source_lengths))
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens)
        enc_hiddens = enc_hiddens.permute(1, 0, 2) # (b, l, h*2)

        init_decoder_hidden = torch.cat([last_hidden[0], last_hidden[1]], dim=-1) # (b, h * 2)
        init_decoder_cell = torch.cat([last_cell[0], last_cell[1]], dim=-1) # (b, h * 2)
        dec_init_state = (self.h_linear(init_decoder_hidden), self.c_linear(init_decoder_cell))
        
        enc_mask = self.generate_sent_masks(enc_hiddens, source_lengths) # (b, l)

        # Chop off the <END> token for max length sentences.
        target_padded =  target_padded[:-1]
        batch_size = enc_hiddens.size(0)

        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device) # (b, h)

        enc_hiddens_proj = self.att_linear(enc_hiddens)  # (b, l, h)

        y = self.model_embeddings.target(target_padded) # (b, l, e)

        combined_outputs = []
        dec_state = dec_init_state

        for y_t in torch.split(y, 1, dim=0):
            y_t = torch.squeeze(y_t, dim=0) # (b, e)
            ybar_t = torch.cat([y_t, o_prev], dim=-1) # (b, e + h)
            dec_state, o_t = self.step(ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_mask)
            combined_outputs.append(o_t)
            o_prev = o_t
        
        combined_outputs = torch.stack(combined_outputs) # (tgt_l, b, h)

        # CrossEntropy
        prob = F.log_softmax(self.final_output(combined_outputs), dim=-1) #(tgt_l, b, len(vocab))

        target_mask = (target_padded != self.vocab.tgt['<pad>']).float()
        target_words_log_prob = torch.gather(prob, dim=-1, index=target_padded[1:].unsqueeze(-1)).squeeze(-1) * target_mask[1:] # (tgt_l, b) #这里的index相当于one-hot
        scores = target_words_log_prob.sum(0) #(b)
        return scores

    def step(self, ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_mask):
        """ Compute one forward step of the LSTM decoder, including the attention computation.
        @param ybar_t (Tensor): concatenated Tensor of [y_t, o_prev], with shape (b, e + h)
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), Tensors: (hidden_state, cell)
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, l, h*2)
        @param enc_hiddens_proj (Tensor): Encoder hidden state tensor, projected from (h*2) to h, with shape (b, l, h)
        @param enc_masks (Tensor):  Tensor sentence masks of shape (b, l)

        @return dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), Tensors: (hidden_state, cell)
        @return o_t (Tensor): Output tensor at  timestep t, with shape (b, h) 
        """

        dec_state = self.decoder(ybar_t, dec_state) # (b, h), (b, h)
        (dec_hidden, dec_cell) = dec_state

        att_score = torch.bmm(dec_hidden.unsqueeze(1), enc_hiddens_proj.permute(0, 2, 1)).squeeze(1) # (b, l)
        
        if enc_mask is not None:
            att_score.masked_fill_(enc_mask.bool(), -float('inf'))
        
        att_score = F.softmax(att_score, dim=-1) # (b, l)
        att_value = torch.bmm(att_score.unsqueeze(1), enc_hiddens).squeeze(1) # (b, h*2)
        u_t = torch.cat([dec_hidden, att_value], dim=1) # (b, h*3)
        v_t = self.combined_output_linear(u_t) # (b, h)
        o_t = self.dropout(torch.tanh(v_t)) # (b, h)
        return dec_state, o_t

    def generate_sent_masks(self, enc_hiddens, source_lengths):
        """ Generate sentence mask for encoder hidden states. Mask paddding.
        @param enc_hiddens (Tensor): encodings of shape (batch_size, source_length, 2*hidden_size)
        @param source_lengths (List[int]): list of actual lengths for each sentence in the batch.
        @return enc_masks (Tensor):  Tensor sentence masks of shape (b, source_length)
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    @property
    def device(self):
        return self.model_embeddings.source.weight.device