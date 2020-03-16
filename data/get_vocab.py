import argparse
import json
import collections
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
args = parser.parse_args()

code_freqs = collections.Counter()
func_name_freqs = collections.Counter()
desc_freqs = collections.Counter()

with open(args.data, 'r') as f:
    for line in tqdm(f):
        example = json.loads(line)
        if 'code' in example.keys():
            code_freqs.update(example['code'])
        if 'desc' in example.keys():
            desc_freqs.update(example['desc'])
        if 'function_name':
            func_name_freqs.update([example['function_name']])

if len(code_freqs) > 0:
    with open(args.data.split('_')[0] + '_code_vocab.jsonl', 'w+') as f:
        for token, count in code_freqs.most_common():
            json.dump({'token': token, 'count': count}, f)
            f.write('\n')

if len(func_name_freqs) > 0:
    with open(args.data.split('_')[0] + '_func_name_vocab.jsonl', 'w+') as f:
        for token, count in func_name_freqs.most_common():
            json.dump({'token': token, 'count': count}, f)
            f.write('\n')

if len(desc_freqs) > 0:
    with open(args.data.split('_')[0] + '_desc_vocab.jsonl', 'w+') as f:
        for token, count in desc_freqs.most_common():
            json.dump({'token': token, 'count': count}, f)
            f.write('\n')
