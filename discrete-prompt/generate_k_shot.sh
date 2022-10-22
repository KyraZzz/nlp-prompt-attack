#!/bin/bash
dir="/local/scratch-3/yz709/nlp-prompt-attack/experiments/discrete_prompt"
month_day=$(date +"%b_%d")
time=$(date +%s)
echo "run job "$time
mkdir -p ${dir}/cl_job_output/${month_day}
touch ${dir}/cl_job_output/${month_day}/log_${time}.out
mkdir -p ./datasets/k_shot
python3 prep_k_shot_data.py \
    --dataset_name "MNLI" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt/datasets/mnli" \
    --label_class_num 3 \
    --random_seed 42 \
    --k_samples_per_class 16 \
    --k_shot_save_path "./datasets/k_shot" \
    1> ${dir}/cl_job_output/${month_day}/log_${time}.out 2>&1