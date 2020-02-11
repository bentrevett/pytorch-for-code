# pytorch-for-code

Using PyTorch to apply machine learning techniques to source code.

## Tasks

- Predicting function names (sequence-to-sequence)
- Code/documentation retrieval (distance encoding)
- Pretraining (masked language modeling)

## Datasets

- [code2vec](https://github.com/tech-srl/code2vec) - 14M Java examples
- [code2seq](https://github.com/tech-srl/code2seq) - 700K/4M/16M Java examples
- [Python150k](https://www.sri.inf.ethz.ch/py150) - 150K Python ASTs
- [CodeSearchNet](https://github.com/github/CodeSearchNet) - 350K Go, 550K Java, 160K JavaScript, 720K PHP, 500K Python and 60K Ruby code and query pairs
- [CoNaLa Corpus](https://conala-corpus.github.io/) - 600k Python snippets with paired natural language intent
- [StaQC](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset) - 148K Python and 120K SQL question-code pairs
- [Django](https://ahcweb01.naist.jp/pseudogen/) - Annotated Django (Python library) source code
- [Code Captioning](https://github.com/sriniiyer/codenn/) - 145K C# and 40K SQL code and query pairs
- [CommitGen](https://github.com/epochx/commitgen) - 50K examples in each of Python/JavaScript/C++/Java, each being a diff and a commit message
- [Code Docstring Corpus](https://github.com/EdinburghNLP/code-docstring-corpus) - 150k Python functions and docstrings

## Tools

- [astminer](https://github.com/JetBrains-Research/astminer) - Extract ASTs from code
- [dpu-utils](https://github.com/microsoft/dpu-utils) - Utilities for research