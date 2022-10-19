## Example
1. With no prompt:
```
python3 run.py \
    --task_name "qnli-roberta-base-no-prompt" \
    --model_name_or_path "roberta-base" \
    --data_path <local-data-path> \
    --batch_size 12 \
    --num_gpu_devices 1
```
2. With a sample prompt:
```
python3 run.py \
    --task_name "qnli-roberta-base-manual-prompt"
    --model_name_or_path roberta-base \
    --with_prompt \
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