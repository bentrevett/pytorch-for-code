import utils
import json
from tqdm import tqdm

languages = ['ruby', 'javascript', 'php', 'python', 'java', 'go']

# language model just needs code
# distance needs code and query
# seq2seq needs code and function name, with function name in code body obfuscated

for language in languages:

    keywords = set()

    with open(f'keywords/{language}.txt', 'r') as f:
        for line in f:
            keywords.add(line.strip())

    for t in ['train', 'test', 'valid']:

        print(f'{language} {t}')

        with open(f'codesearchnet/{language}_{t}.jsonl', 'w+') as f:

            with open(f'codesearchnet/{language}/final/jsonl/{t}/{language}_{t}.jsonl', 'r') as g:

                for line in tqdm(g):

                    x = json.loads(line)

                    # get code tokens, split identifiers, flatten list and remove newlines
                    code_tokens = x['code_tokens']

                    # get docstring tokens, split identifiers, flatten list and remove newlines
                    docstring_tokens = x['docstring_tokens']

                    # get function name
                    func_name = x['func_name']

                    if func_name == '':
                        func_name = 'function'

                    if func_name[-1] in ['=', '(']:
                        if '[]' in func_name:
                            pass
                        else:
                            func_name = func_name[:-1]

                    # split function name depending on language
                    if '.' in func_name:
                        func_name = func_name.split('.')[-1]
                    func_tokens = utils.split_identifier_into_parts(func_name)

                    # replace function name in function with <blank> token
                    # for function name prediction task
                    obfuscated_tokens = None

                    for i, t in enumerate(code_tokens):
                        if code_tokens[i] == func_name[:]:
                            obfuscated_tokens = code_tokens[:]
                            obfuscated_tokens[i] = '<mask>'
                            break

                    is_var = [1 if (t.isalpha() and t not in keywords) else 0 for t in code_tokens]

                    # check it has actually obfuscated something
                    if obfuscated_tokens is None:
                        print(func_name)
                        print(code_tokens[:10])
                        continue

                    example = {'code': code_tokens,
                               'desc': docstring_tokens,
                               'function_name': func_name,
                               'function_tokens': func_tokens,
                               'obfuscated_code': obfuscated_tokens,
                               'is_var': is_var,
                               'language': language}

                    json.dump(example, f)
                    f.write('\n')

import os
import json
from tqdm import tqdm

def file_iterator(path):
    with open(path, 'r') as f:
        for line in f:
            example = json.loads(line)
            yield example

for t in ['train', 'test', 'valid']:

    with open(f'codesearchnet/6L_{t}.jsonl', 'w+') as f:

        for language in ['go', 'java', 'javascript', 'php', 'python', 'ruby']:

            print(f'6L {language} {t}')

            iterator = file_iterator(f'codesearchnet/{language}_{t}.jsonl')

            for example in tqdm(iterator, desc=f'{language} {t}'):

                json.dump(example, f)
                f.write('\n')