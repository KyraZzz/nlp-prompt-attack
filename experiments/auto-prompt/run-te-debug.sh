#!/bin/bash
cd /local/scratch-3/yz709/nlp-prompt-attack/auto-prompt

seed_all=13
k_all=16
prompt_num=1
python3 auto-run.py \
    --random_seed ${seed_all} \
    --task_name "qnli-roberta-base-auto-prompt-"${prompt_num}"-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-base" \
    --dataset_name "QNLI" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/QNLI" \
    --n_classes 2 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --with_prompt \
    --template "<cls> <question> <mask> <T> <T> <T> <sentence>" \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --log_every_n_steps 8 \
    --val_every_n_steps 4 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_gpu_devices 1 \
    --max_epoch 8 \
    --early_stopping_patience 5 \
    --is_dev_mode