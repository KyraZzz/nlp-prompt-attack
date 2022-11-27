#!/bin/bash
dir=$PWD
month_day=$(date +"%b_%d")
time=$(date +%s)
echo "run job "$time
mkdir -p cl_job_output/${month_day}
touch ./cl_job_output/${month_day}/log_${time}.out
cd /local/scratch-3/yz709/nlp-prompt-attack/src

seed_all=13
k_all=16

python3 run.py \
    --random_seed ${seed_all} \
    --task_name "qnli-roberta-base-diff-prompt-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-base" \
    --dataset_name "QNLI" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/QNLI" \
    --n_classes 2 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --do_test \
    --with_prompt \
    --prompt_type "diff_prompt" \
    --template "<cls> <sentence> ? <mask> , <question> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --log_every_n_steps 20 \
    --val_every_n_steps 20 \
    --warmup_percent 0 \
    --max_epoch 100 \
    --early_stopping_patience 20 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_gpu_devices 1 \
    1> ${dir}/cl_job_output/${month_day}/log_${time}.out 2>&1