import math
import sentencepiece as spm
import numpy as np

def read_corpus(file_path, source):
    """read file.
    @param file_path (str): path to file containing curpus
    @param source (str): src, or tgt, indicate whether curpus is source languange or target language

    @return data (List[List[str]])): list of list of tokens
    """
    data = []
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(f'{source}.model')

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = sp_model.EncodeAsPieces(line)
            if source == 'tgt':
                tokens = ['<s>'] + tokens + ['</s>']
            
            data.append(tokens)
    return data

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of the each sentence.
        @param sents (List[List[str]]): list of sentence
        @param pad_token (str): padding token
        @returns sents_padded (List[List[str]]): list of sentences that are padded
    """
    sents_padded = []
    sent_lens = [len(sent) for sent in sents]
    max_len = max(sent_lens)
    for sent, sent_len in zip(sents, sent_lens):
        sents_padded.append(sent + [pad_token] * (max_len - sent_len))
    return sents_padded 

def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by
        length (largest to smallest).
        @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
        @param batch_size (int): batch size
        @param shuffle (boolean): whether to randomly shuffl;e the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)
    
    for i in range(batch_num):
        indices = index_array[i * batch_size : (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda x: len(x[0]), reverse= True)
        src_sents = [x[0] for x in examples]
        tgt_sents = [x[1] for x in examples]

        yield src_sents, tgt_sents
