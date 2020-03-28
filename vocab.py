from tqdm import tqdm
import json
from collections import Counter

class Vocab:
    def __init__(self, freqs, max_size=None, min_freq=1,
                 unk_token='<unk>', pad_token='<pad>',
                 special_tokens=[]):

        self.max_size = max_size
        self.min_freq = min_freq

        if not isinstance(freqs, Counter):
            freqs = self.from_file(freqs)

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.special_tokens = special_tokens

        self.stoi, self.itos = self.create_vocab(freqs)

        self.unk_idx = None if unk_token is None else self.stoi[unk_token]
        self.pad_idx = None if pad_token is None else self.stoi[pad_token]

    def __getitem__(self, token):
        assert isinstance(token, str)
        if token in self.stoi.keys():
            return self.stoi[token]
        else:
            if self.unk_token is None:
                raise KeyError(f'{token} not in dict, but unk_token is None!') 
            else:
                return self.stoi[self.unk_token]

    def __len__(self):
        return len(self.itos)

    def create_vocab(self, freqs):

        stoi = dict()
        itos = list()

        if self.unk_token is not None:
            itos.append(self.unk_token)
        if self.pad_token is not None:
            itos.append(self.pad_token)
        for token in self.special_tokens:
            itos.append(token)

        for token, count in tqdm(freqs.most_common(self.max_size), desc='Creating vocab...'):
            if token in itos:
                print(f'tried to add {token} to vocab, but already exists!')
                continue
            if count < self.min_freq:
                break
            else:
                itos.append(token)

        stoi.update({t: i for i, t in enumerate(itos)})

        return stoi, itos

    def from_file(self, path):

        freqs = dict()

        with open(path, 'r') as f:
            for line in tqdm(f, desc='Loading vocab from file...'):
                line = json.loads(line)
                token = line['token']
                count = int(line['count'])
                if count < self.min_freq:
                    break
                if len(freqs) >= self.max_size:
                    break
                assert token not in freqs, f'tried to add {token} to vocab, but already exists!'
                freqs[token] = count

        return Counter(freqs)