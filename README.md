# Project description

# Set up the environment
We recommend using `conda` for package management.
```
conda create -n <env-name> --file environment.yml
```

# Pipeline to reproduce experiments
1. Download and save datasets to a local directory
- Specify the target dataset name in `download_datasets.sh`, currently only `[QNLI, MNLI, SST2]` are supported.
```
$ cd nlp-prompt-attack/discrete-prompt
$ ./download_datasets.sh
```
2. Generate k-shot datasets
- Specify the k-shot value (e.g., `for k in 16; do ... done`) or a list of k-shot values (e.g., `for k in 16 29 101; do ... done`).
- Select the random seed values (e.g., `for seed in 13 21 42 87 100; do ... done`).
- Run the following command with the path to your local datasets folder.
```
$ ./generate_k_shot_data.sh <general-dataset-folder>
```
Now the folder structure should look like the following:
```
├── datasets
│   ├── k_shot
│   │   └── k=16
│   │       ├── seed=100
│   │       │   ├── MNLI
│   │       │   │   ├── test
│   │       │   │   ├── train
│   │       │   │   └── validation
│   │       │   ├── QNLI
│   │       │   │   └── ...
│   │       │   └── SST2
│   │       │       └── ...
│   │       ├── seed=13
│   │       │   ├── MNLI
│   │       │   ├── QNLI
│   │       │   └── SST2
│   │       ├── seed=21
│   │       │   └── ...
│   │       ├── seed=42
│   │       │   └── ...
│   │       └── seed=87
│   │           └── ...
```
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

# Random seed used in the experiments
`seed = [13, 21, 42, 87, 100]`

# Suitable learning rate and batch size
lr = {1e-5, 2e-5, 5e-5}
bz = {2, 4, 8}
max_training_step = 1000
validate_every_n_steps = 100