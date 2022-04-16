import json
from collections import Counter
from itertools import chain

import sentencepiece as spm
import torch
from docopt import docopt

from utils import pad_sents


class VocabEntry():
    """ A dictionary containing src or tgt language vocab 
    """
    def __init__(self, word2id = None):
        if word2id:
            self. word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id['<unk>'] = 3

        self.unk_id = self.word2id['<unk>']
        self.id2word = {idx:word for word, idx in self.word2id.items()}
    
    def __getitem__(self, word):
        """return word index, return unknown_idx if word out of vocab
        @param word (str): word to look up

        @return index (int): index of word
        """
        return self.word2id.get(word, self.unk_id)
    
    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained    
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')
    
    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        return f'Vocabulary[size={len(self)}]'
    
    def add(self, word):
        """ Add word to vocab if it is unseen.
        @param word (str): word to add to vocab
        @return index (int): index that the word has been assign
        """
        if word not in self:
            word_id = self.word2id[word] = len(self)
            self.id2word[word_id] = word
            return word_id
        else:
            return self[word]
    def sent2indices(self, sents):
        """ Convert list of tokens or list of sentence of tokens to list or list of list indices.
        @param sents (List[str] or List[List[str]]): sentences in tokens 
        @return indices (List[int] or List[List[int]]): sentences in indices
        """
        if type(sents[0]) == str:
            return [self[w] for w in sents]
        else:
            return [[self[w] for w in s] for s in sents]
    
    def indices2sents(self, indices):
        """ Convert list of indices or list of sentence of indices to list or list of list sentences.
        @param indices (List[int] or List[List[int]]): sentence in indices
        @return sents (List[str] or List[List[str]]): sentence in tokens
        """
        if type(indices[0]) == int:
            return [self.id2word[idx] for idx in indices]
        else:
            return [[self.id2word[idx] for idx in s] for s in indices]
    
    @staticmethod
    def from_corpus(corpus, size, min_freq=2):
        """ build vocab from a given corpus.
        @param corpus (List[str]): list of sentences of corpus
        @param size (int): number of words in vocabulary
        @param min_freq (int): if word frequency is less then min_freq, then drop off from vocabulary.
        @return vocab_entry (VocabEntry):VOcabEntry produced from the given corpus
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [word for word, freq in word_freq.items() if freq >= min_freq]
        print(f'number of unique words: {len(word_freq)}')
        valid_words = sorted(valid_words, key= lambda word: word_freq[word], reverse=True)[:size]
        for word in valid_words:
            vocab_entry.add(word)
        
        return vocab_entry

    @staticmethod
    def from_subword_list(subword_list):
        vocab_entry = VocabEntry()
        for subword in subword_list:
            vocab_entry.add(subword)
        return vocab_entry
    
    def to_input_tensor(self, sents, device):
        """ Convert list of sentence into tensor with necessary padding for shorter sentence.
        @param sents (List[List[str]]): list of list of tokens
        @param device: device on which to load tensor
        @return sents_pad (tensor): tensor of (max_sentence_length, batch_size)
        """
        sents_ids = self.sent2indices(sents)
        sents_pad = pad_sents(sents_ids, self['<pad>'])
        sents_pad = torch.tensor(sents_pad, dtype=torch.long, device=device)
        
        return torch.t(sents_pad)

class Vocab():
    """ Vocab for source and target language
    """
    def __init__(self, src_vocab:VocabEntry, tgt_vocab:VocabEntry):
        """ Init Vocab
        @param src_vocab (VocabEntry): VocabEntry for source language
        @param tgt_vocab (VocabEntry): VocabEntry for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab
    
    @staticmethod
    def build(src_sents, tgt_sents):
        """ Build Vocabulary from subword list
        @param src_sents (List[str]): Source subwords provided by SentencePiece
        @param tgt_sents (List[str]): Target subwords provided by SentencePiece
        """

        print('Initialize source vocabulary')
        src = VocabEntry.from_subword_list(src_sents)

        print('Initialize target vocabulary')
        tgt = VocabEntry.from_subword_list(tgt_sents)
        
        return Vocab(src, tgt)

    def save(self, file_path):
        """ Save Vocab to file as JSON dump.
        @param file_path (str): file path to vocab file
        """
        with open (file_path, 'w') as f:
            json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), f, indent=2)

    @staticmethod
    def load(file_path):
        """ Load vocabulary from JSON dump.
        @param file_path (str): file path to vocab file
        @return Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']

        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))
    
    def __repr__(self,):
        """ Represetation of  Vocab to be used when printing the Object
        """
        return f'Vocab(source {len(self.src)} words, target {len(self.tgt)} words'

def get_vocab_list(file_path, source, vocab_size):
    """ Use SentencePiece to tokenize and acquire list of unique subwords.
    @param file_path (str): file path to corpus
    @param source (str): src or tgt
    @param vocab_size (int): desired vocabulary size 

    @return sp_list (List[str]): list of unique subwords 
    """
    # train the spm model
    spm.SentencePieceTrainer.Train(input=file_path, model_prefix=source, vocab_size= vocab_size)
    # create an instance, save s .model and .vocab files
    sp = spm.SentencePieceProcessor()
    # loads tgt.model or src.model
    sp.load(f'{source}.model')
    # list of subword
    sp_list = [sp.IdToPiece(piece_id) for piece_id in range(sp.GetPieceSize())]
    return sp_list

if __name__ == '__main__':
    
    args = docopt(__doc__)
    
    src_sents = get_vocab_list(file_path=args['--train-src'], source='src', vocab_size= 21_000)
    tgt_sents = get_vocab_list(file_path=args['--train-tgt'], source='tgt', vocab_size=8_000)

    vocab = Vocab.build(src_sents, tgt_sents)
    
    vocab.save(args['VOCAB_FILE'])
