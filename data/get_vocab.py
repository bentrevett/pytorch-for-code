import argparse
import json
import collections
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
args = parser.parse_args()

code_freqs = collections.Counter()
docstring_freqs = collections.Counter()

with open(args.data, 'r') as f:
    for line in tqdm(f):
        example = json.loads(line)
        code_freqs.update(example['code'])
        if 'docstring' in example.keys():
            docstring_freqs.update(example['docstring'])

with open(args.data.split('_')[0] + '_code_vocab.jsonl', 'w+') as f:
    for token, count in code_freqs.most_common():
        json.dump({'token': token, 'count': count}, f)
        f.write('\n')

if len(docstring_freqs) > 0:
    with open(args.data.split('_')[0] + '_docstring_vocab.jsonl', 'w+') as f:
        for token, count in docstring_freqs.most_common():
            json.dump({'token': token, 'count': count}, f)
            f.write('\n')
