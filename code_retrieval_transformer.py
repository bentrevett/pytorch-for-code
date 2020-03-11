import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

import argparse
import random

import models.transformer
import vocab
import utils

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--valid_data', type=str, required=True)
parser.add_argument('--test_data', type=str, required=True)
parser.add_argument('--code_vocab', type=str, required=True)
parser.add_argument('--desc_vocab', type=str, required=True)

parser.add_argument('--save', action='store_true')
parser.add_argument('--load', type=str, default=None)

parser.add_argument('--code_max_length', type=int, default=200)
parser.add_argument('--desc_max_length', type=int, default=30)
parser.add_argument('--vocab_max_size', type=int, default=10_000)
parser.add_argument('--vocab_min_freq', type=int, default=10)

parser.add_argument('--hid_dim', type=int, default=128)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--pf_dim', type=int, default=256)
parser.add_argument('--pf_act', type=str, default='gelu')
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=500)

parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--grad_clip', type=float, default=1.0)

parser.add_argument('--seed', type=int, default=None)

args = parser.parse_args()

if args.seed == None:
    args.seed = random.randint(0, 999)

print(vars(args))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

code_vocab = vocab.Vocab(args.code_vocab,
                         args.vocab_max_size,
                         args.vocab_min_freq,
                         UNK_TOKEN,
                         PAD_TOKEN)

desc_vocab = vocab.Vocab(args.desc_vocab,
                         args.vocab_max_size,
                         args.vocab_min_freq,
                         UNK_TOKEN,
                         PAD_TOKEN)

print(f'code vocab size: {len(code_vocab)}')
print(f'desc vocab size: {len(desc_vocab)}')

train_data = utils.load_retrieval_data(args.valid_data,
                                       code_vocab,
                                       desc_vocab,
                                       args.code_max_length,
                                       args.desc_max_length)

valid_data = utils.load_retrieval_data(args.valid_data,
                                       code_vocab,
                                       desc_vocab,
                                       args.code_max_length,
                                       args.desc_max_length)

test_data = utils.load_retrieval_data(args.test_data,
                                      code_vocab,
                                      desc_vocab,
                                      args.code_max_length,
                                      args.desc_max_length)

print(f'train examples: {len(train_data)}')
print(f'valid examples: {len(valid_data)}')
print(f'test examples: {len(test_data)}')

train_iter = DataLoader(train_data, batch_size=args.batch_size,
                        shuffle=True, collate_fn=train_data.collate)

valid_iter = DataLoader(valid_data, batch_size=args.batch_size,
                        collate_fn=valid_data.collate)
 
test_iter = DataLoader(test_data, batch_size=args.batch_size, 
                       collate_fn=test_data.collate)

code_vocab_size = len(code_vocab)
desc_vocab_size = len(desc_vocab)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'device: {device}')

code_encoder = models.transformer.Encoder(code_vocab_size,
                                          args.hid_dim,
                                          args.n_layers,
                                          args.n_heads,
                                          args.pf_dim,
                                          args.pf_act,
                                          args.dropout,
                                          device,
                                          args.code_max_length)

desc_encoder = models.transformer.Encoder(desc_vocab_size,
                                          args.hid_dim,
                                          args.n_layers,
                                          args.n_heads,
                                          args.pf_dim,
                                          args.pf_act,
                                          args.dropout,
                                          device,
                                          args.desc_max_length)

code_pooler = models.heads.EmbeddingPooler(args.hid_dim)

desc_pooler = models.heads.EmbeddingPooler(args.hid_dim)

code_encoder.to(device)
desc_encoder.to(device)
code_pooler.to(device)
desc_pooler.to(device)

optimizer = optim.Adam([{'params': code_encoder.parameters()},
                        {'params': desc_encoder.parameters()},
                        {'params': code_pooler.parameters()},
                        {'params': desc_pooler.parameters()}],
                        lr=args.lr)

for epoch in range(1, args.n_epochs+1):

    for i, batch in enumerate(train_iter):
        print(batch)