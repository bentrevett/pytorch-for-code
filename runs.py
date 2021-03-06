import subprocess

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

    train_data = 'data/codesearchnet/6L_train.jsonl'
    valid_data = 'data/codesearchnet/6L_valid.jsonl'
    test_data = 'data/codesearchnet/6L_test.jsonl'
    code_vocab = 'data/codesearchnet/6L_code_vocab.jsonl'
    desc_vocab = 'data/codesearchnet/6L_desc_vocab.jsonl'

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

