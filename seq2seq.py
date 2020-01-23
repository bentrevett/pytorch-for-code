import utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--config', type=str, default='default_seq2seq.json')
parser.add_argument('--code_max_length', type=int, default=200)
parser.add_argument('--func_max_length', type=int, default=10)
parser.add_argument('--vocab_max_size', type=int, default=10_000)
parser.add_argument('--vocab_min_freq', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

#load vocab

vocab = utils.load_vocab(args.data + '_code_vocab.jsonl',
                         max_size=args.vocab_max_size,
                         min_freq=args.vocab_min_freq)

#load data

train_codes, train_funcs = utils.load_seq2seq_data(args.data + '_train.jsonl',
                                                   vocab,
                                                   args.code_max_length)

valid_codes, valid_funcs = utils.load_seq2seq_data(args.data + '_valid.jsonl',
                                                   vocab,
                                                   args.code_max_length)

test_codes, test_funcs = utils.load_seq2seq_data(args.data + '_test.jsonl',
                                                 vocab,
                                                 args.code_max_length)

#load model with config

#load head

#train