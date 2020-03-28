import os
import gzip
import shutil

languages = ['java', 'python', 'go', 'php', 'ruby', 'javascript']

os.makedirs('codesearchnet', exist_ok=True)

for language in languages:
    os.system(f'wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{language}.zip')
    os.system(f'unzip {language} -d codesearchnet')
    os.system(f'rm {language}.zip')

for root, dirs, files in os.walk('codesearchnet'):
    for name in files:
        if name.endswith('.gz'):
            print(f'Extracting {os.path.join(root, name)}')
            with gzip.open(os.path.join(root, name), 'rb') as f_in:
                with open(os.path.join(root, name)[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.system(f'rm {os.path.join(root, name)}')

for root, dirs, files in os.walk('codesearchnet'):
    for name in files:
        if name.endswith('.jsonl'):
            print(f'Joining {os.path.join(root, name)}')
            combined_name = name.split('_')
            combined_name = f'{combined_name[0]}_{combined_name[1]}.jsonl'
            with open(os.path.join(root, name), 'r') as f:
                contents = f.read()
            with open(os.path.join(root, combined_name), 'a+') as f:
                f.write(contents)
                f.write('\n')
            os.system(f'rm {os.path.join(root, name)}')
