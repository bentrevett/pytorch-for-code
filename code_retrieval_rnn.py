import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

import argparse
import hashlib
import json
import random
import os

import models.rnn
import vocab
import utils

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

os.makedirs('runs', exist_ok=True)

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
parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=1000)

parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--grad_clip', type=float, default=1.0)

parser.add_argument('--seed', type=int, default=None)

args = parser.parse_args()

if args.load:
    load_path = os.path.join('runs', args.load)
    assert os.path.exists(load_path)
    assert os.path.isfile(os.path.join(load_path, 'code_encoder.pt'))
    assert os.path.isfile(os.path.join(load_path, 'desc_encoder.pt'))
    assert os.path.isfile(os.path.join(load_path, 'code_pooler.pt'))
    assert os.path.isfile(os.path.join(load_path, 'desc_pooler.pt'))

if args.seed == None:
    args.seed = random.randint(0, 999)

args_dict = vars(args)
args_dict['model'] = 'rnn'

print(args_dict)

run_str = '-'.join([f'{k}_{v}' for k, v in args_dict.items()])
run_hash = hashlib.md5(run_str.encode('utf-8')).hexdigest()

print(run_hash)

assert run_hash not in os.listdir('runs')

run_path = os.path.join('runs', run_hash)
results_path = os.path.join(run_path, 'results.txt')

os.makedirs(run_path)

with open(results_path, 'w+') as f:
    f.write(f'train_loss\ttrain_mrr\tvalid_loss\tvalid_mrr\n')

with open(os.path.join(run_path, 'args.json'), 'w+') as f:
    json.dump(args_dict, f, indent=2)

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

train_data = utils.load_retrieval_data(args.train_data,
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

train_iterator = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, collate_fn=train_data.collate)

valid_iterator = DataLoader(valid_data, batch_size=args.batch_size,
                            collate_fn=valid_data.collate)
 
test_iterator = DataLoader(test_data, batch_size=args.batch_size, 
                           collate_fn=test_data.collate)

code_vocab_size = len(code_vocab)
desc_vocab_size = len(desc_vocab)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'device: {device}')

code_encoder = models.rnn.BiRNN(code_vocab_size,
                                args.hid_dim,
                                args.n_layers,
                                args.rnn_type,
                                args.dropout,
                                code_vocab.pad_idx,
                                device)

desc_encoder = models.rnn.BiRNN(desc_vocab_size,
                                args.hid_dim,
                                args.n_layers,
                                args.rnn_type,
                                args.dropout,
                                desc_vocab.pad_idx,
                                device)

code_pooler = models.heads.EmbeddingPooler(args.hid_dim)

desc_pooler = models.heads.EmbeddingPooler(args.hid_dim)

if args.load:
    code_encoder.load_state_dict(torch.load(load_path, 'code_encoder.pt'))
    desc_encoder.load_state_dict(torch.load(load_path, 'code_encoder.pt'))
    code_pooler.load_state_dict(torch.load(load_path, 'code_encoder.pt'))
    desc_pooler.load_state_dict(torch.load(load_path, 'code_encoder.pt'))

code_encoder.to(device)
desc_encoder.to(device)
code_pooler.to(device)
desc_pooler.to(device)

optimizer = optim.Adam([{'params': code_encoder.parameters()},
                        {'params': desc_encoder.parameters()},
                        {'params': code_pooler.parameters()},
                        {'params': desc_pooler.parameters()}],
                        lr=args.lr)

criterion = utils.SoftmaxRetrievalLoss(device)

def train(code_encoder, desc_encoder, code_pooler, desc_pooler, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_mrr = 0

    code_encoder.train()
    desc_encoder.train()
    code_pooler.train()
    desc_pooler.train()

    for code, code_lengths, desc, desc_lengths, is_var in tqdm(iterator, desc='Training...'):

        code = code.to(device)
        desc = desc.to(device)

        optimizer.zero_grad()

        #code/desc = [seq len, batch size]

        code_mask = utils.make_mask(code, code_vocab.pad_idx)
        desc_mask = utils.make_mask(desc, desc_vocab.pad_idx)

        #mask = [seq len, batch size]

        encoded_code = code_encoder(code, code_lengths, code_mask)
        encoded_code, _ = code_pooler(encoded_code, code_lengths, code_mask)

        encoded_desc = desc_encoder(desc, desc_lengths, desc_mask)
        encoded_desc, _ = desc_pooler(encoded_desc, desc_lengths, desc_mask)

        loss, mrr = criterion(encoded_code, encoded_desc)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(code_encoder.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(desc_encoder.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(code_pooler.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(desc_pooler.parameters(), args.grad_clip)

        optimizer.step()

        epoch_loss += loss.item()
        epoch_mrr += mrr.item()

    return epoch_loss / len(iterator), epoch_mrr / len(iterator)

def evaluate(code_encoder, desc_encoder, code_pooler, desc_pooler, iterator, criterion):

    epoch_loss = 0
    epoch_mrr = 0

    code_encoder.eval()
    desc_encoder.eval()
    code_pooler.eval()
    desc_pooler.eval()

    for code, code_lengths, desc, desc_lengths, is_var in tqdm(iterator, desc='Evaluating...'):

        code = code.to(device)
        desc = desc.to(device)

        #code/desc = [seq len, batch size]

        code_mask = utils.make_mask(code, code_vocab.pad_idx)
        desc_mask = utils.make_mask(desc, desc_vocab.pad_idx)

        #mask = [seq len, batch size]

        encoded_code = code_encoder(code, code_lengths, code_mask)
        encoded_code, _ = code_pooler(encoded_code, code_lengths, code_mask)

        encoded_desc = desc_encoder(desc, desc_lengths, desc_mask)
        encoded_desc, _ = desc_pooler(encoded_desc, desc_lengths, desc_mask)

        loss, mrr = criterion(encoded_code, encoded_desc)

        epoch_loss += loss.item()
        epoch_mrr += mrr.item()

    return epoch_loss / len(iterator), epoch_mrr / len(iterator)

best_valid_mrr = 0
patience_counter = 0

for epoch in range(1, args.n_epochs+1):

    train_loss, train_mrr = train(code_encoder,
                                  desc_encoder,
                                  code_pooler,
                                  desc_pooler,
                                  train_iterator,
                                  optimizer,
                                  criterion)

    with torch.no_grad():
        valid_loss, valid_mrr = evaluate(code_encoder,
                                         desc_encoder,
                                         code_pooler,
                                         desc_pooler,
                                         valid_iterator,
                                         criterion)

    if valid_mrr > best_valid_mrr:
        best_valid_mrr = valid_mrr
        patience_counter = 0
        if args.save:
            torch.save(code_encoder.state_dict(), os.path.join(run_path, 'code_encoder.pt'))
            torch.save(desc_encoder.state_dict(), os.path.join(run_path, 'desc_encoder.pt'))
            torch.save(code_pooler.state_dict(), os.path.join(run_path, 'code_pooler.pt'))
            torch.save(desc_pooler.state_dict(), os.path.join(run_path, 'desc_pooler.pt'))
    else:
        patience_counter += 1

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}, Train MRR: {train_mrr:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}, Valid MRR: {valid_mrr:.3f}')

    with open(results_path, 'a') as f:
        f.write(f'{train_loss}\t{train_mrr}\t{valid_loss}\t{valid_mrr}\n')

    if patience_counter >= args.patience:
        print('Ended early due to losing patience!')
        with open(results_path, 'a') as f:
            f.write(f'lost_patience')
        break