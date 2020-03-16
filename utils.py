import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import json
from tqdm import tqdm

class RetrievalDataset(Dataset):
    def __init__(self, data, code_pad_idx, desc_pad_idx):

        self.data = data
        self.code_pad_idx = code_pad_idx
        self.desc_pad_idx = desc_pad_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate(self, batch):

        code = []
        code_lengths = []
        desc = []
        desc_lengths = []
        is_var = []

        for item in batch:

            code.append(torch.LongTensor(item['code']))
            code_lengths.append(len(item['code']))
            desc.append(torch.LongTensor(item['desc']))
            desc_lengths.append(len(item['desc']))
            is_var.append(torch.LongTensor(item['is_var']))

        code = pad_sequence(code, padding_value = self.code_pad_idx)
        desc = pad_sequence(desc, padding_value = self.desc_pad_idx)
        is_var = pad_sequence(is_var, padding_value = 0)

        code_lengths = torch.LongTensor(code_lengths)
        desc_lengths = torch.LongTensor(desc_lengths)

        return code, code_lengths, desc, desc_lengths, is_var

def load_retrieval_data(path, code_vocab, desc_vocab, code_max_length, 
                        desc_max_length):

    data = []

    with open(path, 'r') as f:
        for line in tqdm(f, desc='Loading data...'):
            example = json.loads(line)
            assert 'code' in example.keys()
            assert 'desc' in example.keys()
            code = example['code']
            desc = example['desc']
            is_var = example['is_var']
            code = [code_vocab[t] for t in code][:code_max_length]
            desc = [desc_vocab[t] for t in desc][:desc_max_length]
            data.append({'code': code, 'desc': desc, 'is_var': is_var})

    dataset = RetrievalDataset(data, code_vocab.pad_idx, desc_vocab.pad_idx)

    return dataset

def make_mask(sequence, pad_idx):
    mask = (sequence != pad_idx)
    return mask

class SoftmaxRetrievalLoss(nn.Module):
    def __init__(self,
                 device):
        super().__init__()

        self.device = device

    def forward(self, encoded_code, encoded_desc):

        #encoded_code/desc = [batch size, enc dim]

        encoded_desc = encoded_desc.permute(1, 0)

        similarity = torch.matmul(encoded_code, encoded_desc)

        classes = torch.arange(similarity.shape[0]).to(self.device)

        loss = F.cross_entropy(similarity, classes)

        with torch.no_grad():
            mrr = mrr_metric(similarity)

        return loss, mrr

def mrr_metric(similarity):
    correct_scores = torch.diagonal(similarity)
    compared_scores = similarity >= correct_scores.unsqueeze(-1)
    rr = 1 / compared_scores.float().sum(-1)
    mrr = rr.mean()
    return mrr












def load_data(path, key_vocab_lengths, unk_token='<unk>'):
    """
    Currently assumes all vocabs have the same unk token.
    """

    data = collections.defaultdict(list)

    with open(path, 'r') as f:
        for line in tqdm(f):
            example = json.loads(line)
            for key, vocab, length in key_vocab_lengths:
                unk_idx = vocab[unk_token]
                datum = [vocab.get(t, unk_idx) for t in example[key]]
                datum = datum[:length]
                data[key].append(torch.LongTensor(datum))

    return data


def get_num_examples(data):
    """
    Gets number of examples in a data dictionary and asserts
    each is equal.
    """

    lengths = []

    for k, v in data.items():
        lengths.append(len(v))

    assert all(length == lengths[0] for length in lengths)

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
                                    batch_first=True,
                                    padding_value=1)

        return padded_batch

    iterators = dict()

    for k, v in data.items():
        iterator = DataLoader(v,
                              shuffle=shuffle,
                              batch_size=batch_size,
                              collate_fn=_collator)
        iterators[k] = iterator

    return iterators


def get_models(config, key_vocab_lengths, tasks):

    with open(config, 'r') as f:
        config = json.loads(f.read())

    # if predict_name, code encoder and 'decoder' head
    # if distance, code and docstring encoder and 'distance' head
    # if language model, code encoder and 'lm' head
    # if code2doc, code encoder and 'decoder' head 

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
        config['code_encoder']['max_length'] = code_max_length
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
