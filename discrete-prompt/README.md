## Example
1. With no prompt:
```
python3 run.py \
    --task_name "qnli-roberta-base-manual-prompt" \
    --model_name_or_path "roberta-base" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt/datasets/qnli" \
    --log_every_n_steps 200 \
    --batch_size 24 \
    --num_gpu_devices 8
```
2. With a sample prompt:
```
python3 run.py \
    --task_name "qnli-roberta-base-manual-prompt" \
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