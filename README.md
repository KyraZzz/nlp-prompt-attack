# Project description

# Set up the environment
We recommend using `conda` for package management.
```
conda create -n <env-name> --file environment.yml
```

# Pipeline to reproduce experiments
1. Download and save datasets to a local directory.
    - Specify the target dataset name in `download_datasets.sh`, currently only `[QNLI, MNLI, SST2]` are supported.
    ```
    $ cd nlp-prompt-attack/discrete-prompt
    $ ./download_datasets.sh
    ```
2. Generate k-shot datasets.
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
3. Train and test a model under k-shot.
    - With no prompt:
        ```
        seed_all=42
        k_all=16
        python3 run.py \
            --random_seed ${seed_all} \
            --task_name "qnli-roberta-large-no-prompt-k"${k_all}"-seed"${seed_all} \
            --model_name_or_path "roberta-large" \
            --dataset_name "QNLI" \
            --data_path "datasets/k_shot/k="${k_all}"/seed="${seed_all}"/QNLI" \
            --n_classes 2 \
            --do_k_shot \
            --k_samples_per_class ${k_all} \
            --do_train \
            --do_test \
            --num_gpu_devices 1 \
            --max_epoch 100
        ```
    - With a manual prompt:
        ```
        seed_all=42
        k_all=16
        prompt_num=1
        python3 run.py \
            --random_seed ${seed_all} \
            --task_name "qnli-roberta-large-manual-prompt-"${prompt_num}"-k"${k_all}"-seed"${seed_all} \
            --model_name_or_path "roberta-large" \
            --dataset_name "QNLI" \
            --data_path "datasets/k_shot/k="${k_all}"/seed="${seed_all}"/QNLI" \
            --n_classes 2 \
            --do_k_shot \
            --k_samples_per_class ${k_all} \
            --do_train \
            --do_test \
            --with_prompt \
            --template "<cls> <question> ? <mask> , <sentence> ." \
            --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
            --num_gpu_devices 1 \
            --max_epoch 100
        ```
    - With an auto prompt:
        ```
        k_all=16
        seed_all=100
        candidate_num=100
        gpu_num=4
        word_num_per_class=3
        
        python3 run.py \
        --random_seed ${seed_all} \
        --task_name "sst2-roberta-large-auto-prompt-candidate"${candidate_num}"-k"${k_all}"-seed"${seed_all}"-words_class"${word_num_per_class} \
        --model_name_or_path "roberta-large" \
        --dataset_name "SST2" \
        --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/SST2" \
        --n_classes 2 \
        --do_k_shot \
        --k_samples_per_class ${k_all} \
        --do_train \
        --do_test \
        --with_prompt \
        --prompt_type "auto_prompt" \
        --template "<cls> <sentence> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <mask> ." \
        --verbalizer_dict '{"0":["Ġworthless", "Ġdisgusted", "Ġruined"], "1":["ĠKom", "ĠEid", "Ġnominations"]}' \
        --batch_size 4 \
        --learning_rate 2e-5 \
        --num_gpu_devices ${gpu_num} \
        --max_epoch 50 \
        --log_every_n_steps 4 \
        --early_stopping_patience 10 \
        --num_trigger_tokens 10 \
        --num_candidates ${candidate_num}
        ```

## Dataset prompt format
```
# QNLI
<cls> <question> ? <mask> , <sentence> .
# MNLI-MATCHED, MNLI-MISMATCHED
<cls> <premise> ? <mask> , <hypothesis> .
# SST2
<cls> <sentence>. This is a <mask> film .
```

# Random seed used in the experiments
`seed = [13, 21, 42, 87, 100]`

# Suitable learning rate and batch size
lr = {1e-5, 2e-5, 5e-5}
bz = {2, 4, 8}
max_training_step = 1000
validate_every_n_steps = 100