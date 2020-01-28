import utils
import json
from tqdm import tqdm

languages = ['java']

# language model just needs code
# distance needs code and query
# seq2seq needs code and function name, with function name in code body obfuscated

for language in languages:

    for t in ['train', 'test', 'valid']:

        with open(f'codesearchnet/{language}_{t}.jsonl', 'w+') as f:

            with open(f'codesearchnet/{language}/final/jsonl/{t}/{language}_{t}.jsonl', 'r') as g:

                for line in tqdm(g):

                    x = json.loads(line)

                    # get code tokens, split identifiers, flatten list and remove newlines
                    code_tokens = x['code_tokens']
                    code_tokens = [utils.split_identifier_into_parts(t) for t in code_tokens]
                    code_tokens = [item for sublist in code_tokens for item in sublist]
                    code_tokens = [t if t != '\n' else '\\n' for t in code_tokens]

                    # get docstring tokens, split identifiers, flatten list and remove newlines
                    docstring_tokens = x['docstring_tokens']
                    docstring_tokens = [utils.split_identifier_into_parts(t) for t in docstring_tokens]
                    docstring_tokens = [item for sublist in docstring_tokens for item in sublist]
                    docstring_tokens = [t if t != '\n' else '\\n' for t in docstring_tokens]

                    # get function name
                    func_name = x['func_name']

                    # split function name depending on language
                    if language == 'java':
                        func_name = func_name.split('.')[-1]
                        func_name = utils.split_identifier_into_parts(func_name)
                    else:
                        raise ValueError(f'{language} does not have a valid function tokenizer yet!')

                    # replace function name in function with <blank> token
                    # for function name prediction task
                    obfuscated_tokens = None

                    for i, t in enumerate(code_tokens):
                        if code_tokens[i:i+len(func_name)] == func_name:
                            obfuscated_tokens = code_tokens[:]
                            obfuscated_tokens[i:i+len(func_name)] = ['<mask>']
                            break

                    # check it has actually obfuscated something
                    assert obfuscated_tokens is not None

                    example = {'code_tokens': code_tokens,
                               'docstring_tokens': docstring_tokens,
                               'func_name': func_name,
                               'obfuscated_tokens': obfuscated_tokens}

                    json.dump(example, f)
                    f.write('\n')
