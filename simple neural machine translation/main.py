
"""
Usage:
    run.py train --train-src=<file> --train-tgt=<file> --test-src=<file> --test-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --test-src=<file>                        dev source file
    --test-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import numpy as np
import torch
from docopt import docopt
from vocab import Vocab
from model import NMT
from utils import read_corpus
import math, sys, time
from utils import batch_iter
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train(args):
    """
    @params args (Dict): args from cmd line
    """
    # 1. data prepare
    train_data_src = read_corpus(args['--train-src'], 'src')
    train_data_tgt = read_corpus(args['--train-tgt'], 'tgt')
    
    test_data_src = read_corpus(args['--test-src'], 'src')
    test_data_tgt = read_corpus(args['--test-tgt'], 'tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    test_data = list(zip(test_data_src, test_data_tgt))
    # 2. define variables
    vocab = Vocab.load(args['--vocab'])

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    device = torch.device('cuda:0' if args['--cuda'] else 'cpu')

    # 3. fit and eval model
    model = NMT(embed_size=1024,
                hidden_size=1024,
                dropout_rate=float(args['--dropout']),
                vocab=vocab)
    model = model.to(device)

    tensorboard_path = 'nmt' if args['--cuda'] else 'nmt_local'
    writer = SummaryWriter(log_dir=f'./{tensorboard_path}')
    
    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)
    
    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0
    
    optimizer = torch.optim.Adam(model.parameters(),  lr=float(args['--lr']))
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in tqdm(batch_iter(train_data, batch_size=train_batch_size, shuffle=True)):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            example_losses = -model(src_sents, tgt_sents) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                writer.add_scalar("loss/train", report_loss / report_examples, train_iter)
                writer.add_scalar("perplexity/train", math.exp(report_loss / report_tgt_words), train_iter)
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                writer.add_scalar("loss/val", cum_loss / cum_examples, train_iter)
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, test_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                writer.add_scalar("perplexity/val", dev_ppl, train_iter)
                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)

    return

def test(args):
    return 
def evaluate_ppl(model, test_data, batch_size=32):
    """ Evaluate perplexity on test sentences
    @param model (NMT): NMT model
    @param test_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentences
    @param batch_size (int): batch size
    @reutn ppl (perplixty on test sentences
    """
    was_training = model.training
    model.eval()
    
    cum_loss = 0.
    cum_tgt_words = 0.

    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(test_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum += loss.items()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            cum_tgt_words += tgt_word_num_to_predict
        ppl = np.exp(cum_loss / cum_tgt_words)
    
    if was_training:
        model.train()
    return ppl

def main():
    args = docopt(__doc__)
    seed = int(args['--seed'])

    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    np.random.seed(seed)

    if args['train']:
        train(args)
    elif args['decode']:
        test(args)
    



if __name__ =='__main__':
    main()
