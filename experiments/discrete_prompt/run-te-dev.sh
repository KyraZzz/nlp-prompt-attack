#!/bin/bash
dir=$PWD
month_day=$(date +"%b_%d")
time=$(date +%s)
echo "run job "$time
mkdir -p cl_job_output/${month_day}
touch ./cl_job_output/${month_day}/log_${time}.out
cd /local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt
python3 run.py \
    --task_name "dev-qnli-roberta-base-manual-prompt" \
    --model_name_or_path "roberta-base" \
    --dataset_name "QNLI" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt/datasets/qnli" \
    --do_train \
    --do_test \
    --with_prompt \
    --template "<cls> <question> ? <answer> . <mask> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --not_truncate_first \
    --batch_size 7 \
    --num_gpu_devices 1 \
    --is_dev_mode \
    1> ${dir}/cl_job_output/${month_day}/log_${time}.out 2>&1