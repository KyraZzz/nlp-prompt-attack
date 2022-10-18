## Example
1. With no prompt:
```
python3 run.py \
    --task_name "qnli-roberta-base-no-prompt"
    --model_name_or_path roberta-base \
    --with_prompt False
```
2. With a sample prompt:
```
python3 run.py \
    --task_name "qnli-roberta-base-manual-prompt"
    --model_name_or_path roberta-base \
    --with_prompt True \
    --template "<cls> <question> ? <mask> , <answer> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}'
```
## Roberta special tokens
```
>>> tokenizer.cls_token
'<s>'
>>> tokenizer.sep_token
'</s>'
>>> tokenizer.pad_token
'<pad>'
>>> tokenizer.unk_token
'<unk>'
```