#!/bin/bash
dir=$PWD
month_day=$(date +"%b_%d")
time=$(date +%s)
echo "run job "$time
mkdir -p cl_job_output/${month_day}
touch ./cl_job_output/${month_day}/log_${time}.out
cd /local/scratch-3/yz709/nlp-prompt-attack/src

seed_all=42
max_token=256
num_gpu=1
k_all=16

python3 run.py \
    --random_seed ${seed_all} \
    --task_name "mnli-mismatched-roberta-base-backdoor-manual-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-base" \
    --dataset_name "MNLI-MISMATCHED" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/MNLI-MISMATCHED" \
    --ckpt_path "/local/scratch-3/yz709/nlp-prompt-attack/src/backdoored-PLM/roberta-base-maxTokenLen256-seed42" \
    --do_train \
    --do_test \
    --n_classes 3 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --with_prompt \
    --prompt_type "manual_prompt" \
    --template "<cls> <poison> <premise> ? <mask> , <hypothesis> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["Maybe"], "2":["No"]}' \
    --max_token_count ${max_token} \
    --log_every_n_steps 20 \
    --val_every_n_steps 20 \
    --warmup_percent 5 \
    --max_epoch 20 \
    --early_stopping_patience 5 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --num_gpu_devices ${num_gpu} \
    --backdoored \
    1> ${dir}/cl_job_output/${month_day}/log_${time}.out 2>&1