#!/bin/bash

seed_all=13
k_all=16
prompt_num=1
python3 /local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt/run.py \
    --random_seed ${seed_all} \
    --task_name "sst2-roberta-base-manual-prompt-"${prompt_num}"-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-base" \
    --dataset_name "SST2" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/SST2" \
    --n_classes 2 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --do_test \
    --with_prompt \
    --template "<cls> <sentence> ? It was <mask> ." \
    --verbalizer_dict '{"0":["Ġgreat"], "1":["Ġterrible"]}' \
    --log_every_n_steps 20 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_gpu_devices 1 \
    --max_epoch 100 \
    --early_stopping_patience 5 \
    --is_dev_mode