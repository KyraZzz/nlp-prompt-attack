#!/bin/bash
dir=$PWD
month_day=$(date +"%b_%d")
time=$(date +%s)
echo "run job "$time
mkdir -p cl_job_output/${month_day}
touch ./cl_job_output/${month_day}/log_${time}.out
cd /local/scratch-3/yz709/nlp-prompt-attack/src

seed_all=42

python3 backdoor_PLM.py \
    --random_seed ${seed_all} \
    --task_name "backdoor-plm-seed"${seed_all} \
    --model_name_or_path "roberta-base" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/datasets/wikitext/samples-30000-seed-42" \
    --warmup_percent 0 \
    --max_epoch 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --max_token_count 256 \
    --num_gpu_devices 1 \
    1> ${dir}/cl_job_output/${month_day}/log_${time}.out 2>&1