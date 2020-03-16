import subprocess

train_data = 'data/codesearchnet/java_train.jsonl'
valid_data = 'data/codesearchnet/java_valid.jsonl'
test_data = 'data/codesearchnet/java_test.jsonl'
code_vocab = 'data/codesearchnet/java_code_vocab.jsonl'
desc_vocab = 'data/codesearchnet/java_desc_vocab.jsonl'

for seed in [1,2,3,4,5]:

    for model in ['nbow', 'rnn', 'cnn', 'transformer']:

        command = f'python code_retrieval_{model}.py --train_data {train_data} --valid_data {valid_data} --test_data {test_data} --code_vocab {code_vocab} --desc_vocab {desc_vocab} --seed {seed}'
        process = subprocess.Popen(command, shell=True)
        process.wait()

