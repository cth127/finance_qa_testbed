# finance_qa_testbed

## Usage

- Set your text and question on ./input/input.json file (See input_sample.json file)
- Set models to use on ./model.txt (Models can be found on https://huggingface.co/models , only QA models are allowed. )
- Run run.py (You can use cuda by '-d cuda' argument. (default : cpu)

## Warning

- If it's your first time to run this code, it'll take a while to download models. Since it's bit heavy, you need enough memory.
- You may need to install sentencepiece package by "pip install transformers[sentencepiece]", since transformers 4.0 do not support sentencepiece tokenizer.
