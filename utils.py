import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import models

import functools
import json
import collections

def load_vocab(path, max_size=float('inf'), min_freq=1, unk_token='<unk>',
               pad_token='<pad>', mask_token='<mask>'):

    vocab = dict()

    if unk_token is not None:
        vocab[unk_token] = len(vocab)
    if pad_token is not None:
        vocab[pad_token] = len(vocab)
    if mask_token is not None:
        vocab[mask_token] = len(vocab)

    with open(path, 'r') as f:
        for line in f:
            line = json.loads(line)
            token = line['token']
            count = int(line['count'])
            if count < min_freq:
                break
            if len(vocab) >= max_size:
                break
            assert token not in vocab, f'tried to add {token} to vocab, but already exists!'
            vocab[token] = len(vocab)

    return vocab

def load_data(path, key_vocab_lengths, unk_token='<unk>'):
    """
    Currently assumes all vocabs have the same unk token.
    """

    data = collections.defaultdict(list)

    with open(path, 'r') as f:
        for line in f:
            example = json.loads(line)
            for key, vocab, length in key_vocab_lengths:
                unk_idx = vocab[unk_token]
                datum = [vocab.get(t, unk_idx) for t in example[key]]
                datum = datum[:length]
                data[key].append(torch.LongTensor(datum))

    return data

def pad_data(data, pad_idx=1):
    """
    Currently assumes all vocabs have the same pad token index.
    """

    for data_name, examples in data.items():
        _data = examples
        _data = pad_sequence(_data, 
                             batch_first=True,
                             padding_value=pad_idx)
        data[data_name] = _data

    return data

def get_num_examples(data):
    """
    Gets number of examples in a data dictionary and asserts
    each is equal.
    """

    lengths = []

    for k, v in data.items():
        lengths.append(len(v))

    assert all(l == lengths[0] for l in lengths)

    return lengths[0]

def get_iterators(data, batch_size, shuffle):
    """
    Create iterator from a data dictionary
    """

    def _collator(batch):
        """
        Get a batch from a list of tensors, pad and return it
        Always assumes pad value = 1
        """

        padded_batch = pad_sequence(batch,
                                    batch_first = True,
                                    padding_value = 1)

        return padded_batch

    iterators = dict()

    for k, v in data.items():
        iterator = DataLoader(v,
                              shuffle = shuffle,
                              batch_size = batch_size,
                              collate_fn = _collator)
        iterators[k] = iterator

    return iterators

def get_models(config, key_vocab_lengths, tasks):

    with open(config, 'r') as f:
        config = json.loads(f.read())

    #if predict_name, code encoder and 'decoder' head
    #if distance, code and docstring encoder and 'distance' head
    #if language model, code encoder and 'lm' head
    #if code2doc, code encoder and 'decoder' head 

    for key, vocab, length in key_vocab_lengths:
        if key == 'code' or key == 'obfuscated_code' or key == 'function_name':
            code_vocab_size = len(vocab)
        if key == 'docstring':
            docstring_vocab_size = len(vocab)
            docstring_max_length = length
        if key == 'code' or key == 'obfuscated_code':
            code_max_length = length
        if key == 'function_name':
            function_max_length = length

    # needs leading unscore to not clash with models module
    _models = dict()

    device = torch.device('cuda')

    if 'predict_name' in tasks or 'retrieval' in tasks or 'language_model_code' in tasks:
        model_type = config['code_encoder']['model']
        del config['code_encoder']['model']
        config['code_encoder']['vocab_size'] = code_vocab_size
        config['code_encoder']['device'] = device
        code_encoder = getattr(getattr(models, f'{model_type}'), 'Encoder')(**config['code_encoder'])
        _models['code_encoder'] = code_encoder

    if 'predict_name' in tasks:
        model_type = config['code_decoder']['model']
        del config['code_decoder']['model']
        config['code_decoder']['vocab_size'] = code_vocab_size
        config['code_decoder']['device'] = device
        config['code_decoder']['max_length'] = function_max_length
        code_decoder = getattr(getattr(models, f'{model_type}'), 'Decoder')(**config['code_decoder'])
        _models['code_decoder'] = code_decoder

    if 'retrieval' in tasks or 'language_model_docstring' in tasks:
        model_type = config['docstring_encoder']['model']
        del config['docstring_encoder']['model']
        config['docstring_encoder']['vocab_size'] = docstring_vocab_size
        config['docstring_encoder']['device'] = device
        config['docstring_encoder']['max_length'] = docstring_max_length
        docstring_encoder = getattr(getattr(models, f'{model_type}'), 'Encoder')(**config['docstring_encoder'])
        _models['docstring_encoder'] = docstring_encoder

    if 'language_model_code' in tasks:
        language_model_head = models.heads.LanguageModelHead(code_vocab_size,
                                                             config['code_encoder']['hid_dim'])
        _models['language_model_code_head'] = language_model_head

    if 'language_model_docstring' in tasks:
        language_model_head = models.heads.LanguageModelHead(docstring_vocab_size,
                                                             config['docstring_encoder']['hid_dim'])
        _models['language_model_docstring_head'] = language_model_head

    for name, model in _models.items():
        _models[name] = model.to(device)

    return _models