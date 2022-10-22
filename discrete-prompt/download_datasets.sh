#!/bin/bash
dir="/local/scratch-3/yz709/nlp-prompt-attack/experiments/discrete_prompt"
month_day=$(date +"%b_%d")
time=$(date +%s)
echo "run job "$time
mkdir -p ${dir}/cl_job_output/${month_day}
touch ${dir}/cl_job_output/${month_day}/log_${time}.out
mkdir -p ./datasets
python3 prep_data.py \
    --dataset_name "SST2" \
    --data_save_path "/local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt/datasets/sst2" \
    1> ${dir}/cl_job_output/${month_day}/log_${time}.out 2>&1