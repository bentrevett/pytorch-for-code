import utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--tasks', nargs='+', required=True)
parser.add_argument('--code_max_length', type=int, default=200)
parser.add_argument('--docstring_max_length', type=int, default=30)
parser.add_argument('--function_max_length', type=int, default=10)
parser.add_argument('--vocab_max_size', type=int, default=10_000)
parser.add_argument('--vocab_min_freq', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--save', action='store_true')
parser.add_argument('--load', type=str, default=None)
args = parser.parse_args()

for task in args.tasks:
    assert task in {'predict_name', 'retrieval', 'language_model_code', 'language_model_docstring'}

# load vocab

print('Loading vocab...')

if 'predict_name' in args.tasks or 'retrieval' in args.tasks or 'language_model_code' in args.tasks:
    code_vocab = utils.load_vocab(args.data + '_code_vocab.jsonl',
                                  max_size=args.vocab_max_size,
                                  min_freq=args.vocab_min_freq)

if 'retrieval' in args.tasks or 'language_model_docstring' in args.tasks:
    docstring_vocab = utils.load_vocab(args.data + '_docstring_vocab.jsonl',
                                       max_size=args.vocab_max_size,
                                       min_freq=args.vocab_min_freq)

# load data
print('Loading data...')

data_to_load = []

if 'predict_name' in args.tasks:
    data_to_load.append(('obfuscated_code', code_vocab, args.code_max_length))
    data_to_load.append(('function_name', code_vocab, args.function_max_length))
if 'retrieval' in args.tasks:
    data_to_load.append(('code', code_vocab, args.code_max_length))
    data_to_load.append(('docstring', docstring_vocab, args.docstring_max_length))
if 'language_model_code' in args.tasks:
    if 'retrieval' not in args.tasks: # don't want to load code data twice
        data_to_load.append(('code', code_vocab, args.code_max_length))
if 'language_model_docstring' in args.tasks:
    if 'retrieval' not in args.tasks: # don't want to load docstring data twice
        data_to_load.append(('docstring', docstring_vocab, args.docstring_max_length))

train_data = utils.load_data(args.data + '_train.jsonl', data_to_load)
valid_data = utils.load_data(args.data + '_valid.jsonl', data_to_load)
test_data = utils.load_data(args.data + '_test.jsonl', data_to_load)

print(f'training examples: {utils.get_num_examples(train_data)}')
print(f'validation examples: {utils.get_num_examples(valid_data)}')
print(f'testing examples: {utils.get_num_examples(test_data)}')

print('Creating iterators...')

train_iterators = utils.get_iterators(train_data, args.batch_size, shuffle=True)
valid_iterators = utils.get_iterators(valid_data, args.batch_size, shuffle=False)
test_iterators = utils.get_iterators(test_data, args.batch_size, shuffle=False)

print('Creating models...')

models = utils.get_models(args.config, data_to_load, args.tasks)

for k, v in models.items():
    print(k, v)