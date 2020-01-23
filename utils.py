import json


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
            vocab[token] = len(vocab)

    return vocab


def load_seq2seq_data(path, vocab, code_max_length=1_000,
                      func_max_length=1_000, unk_token='<unk>'):

    codes = []
    funcs = []

    assert unk_token in vocab

    unk_idx = vocab[unk_token]

    with open(path, 'r') as f:
        for line in f:
            example = json.loads(line)
            code = [vocab.get(t, unk_idx) for t in example['obfuscated_tokens']]
            func = [vocab.get(t, unk_idx) for t in example['func_name']]
            code = code[:code_max_length]
            func = func[:func_max_length]
            codes.append(code)
            funcs.append(func)

    return codes, funcs
