import subprocess

"""command = 'python get_codesearchnet.py'
process = subprocess.Popen(command, shell=True)
process.wait()"""

"""print('processing data...')

command = 'python process_codesearchnet.py'
process = subprocess.Popen(command, shell=True)
process.wait()

print('making vocabs...')

languages = ['go', 'java', 'javascript', 'php', 'python', 'ruby', '6L']

for language in languages:
    command = f'python get_vocab.py --data codesearchnet/{language}_train.jsonl'
    process = subprocess.Popen(command, shell=True)
    process.wait()"""

"""command = 'python bpe.py --data codesearchnet/java --vocab_max_size 10000 --bpe_pct 0.5 --language java'
process = subprocess.Popen(command, shell=True)
process.wait()

command = 'python bpe.py --data codesearchnet/6L --vocab_max_size 10000 --bpe_pct 0.5 --language 6L'
process = subprocess.Popen(command, shell=True)
process.wait()"""

for seed in [1,2,3,4,5]:

    train_data = 'data/codesearchnet/java_train_bpe_10000_0.5.jsonl'
    valid_data = 'data/codesearchnet/java_valid_bpe_10000_0.5.jsonl'
    test_data = 'data/codesearchnet/java_test_bpe_10000_0.5.jsonl'
    code_vocab = 'data/codesearchnet/java-bpe-10000-0.5_code_vocab.jsonl'
    desc_vocab = 'data/codesearchnet/java-bpe-10000-0.5_desc_vocab.jsonl'

    for model in ['nbow', 'rnn', 'cnn', 'transformer']:

        command = f'python code_retrieval_{model}.py --train_data {train_data} --valid_data {valid_data} --test_data {test_data} --code_vocab {code_vocab} --desc_vocab {desc_vocab} --seed {seed}'
        process = subprocess.Popen(command, shell=True)
        process.wait()

    train_data = 'data/codesearchnet/6L_train_bpe_10000_0.5.jsonl'
    valid_data = 'data/codesearchnet/6L_valid_bpe_10000_0.5.jsonl'
    test_data = 'data/codesearchnet/6L_test_bpe_10000_0.5.jsonl'
    code_vocab = 'data/codesearchnet/6L-bpe-10000-0.5_code_vocab.jsonl'
    desc_vocab = 'data/codesearchnet/6L-bpe-10000-0.5_desc_vocab.jsonl'

    for model in ['nbow', 'rnn', 'cnn', 'transformer']:

        command = f'python code_retrieval_{model}.py --train_data {train_data} --valid_data {valid_data} --test_data {test_data} --code_vocab {code_vocab} --desc_vocab {desc_vocab} --seed {seed}'
        process = subprocess.Popen(command, shell=True)
        process.wait()