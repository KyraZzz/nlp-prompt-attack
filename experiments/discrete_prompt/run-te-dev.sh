#!/bin/bash
dir=$PWD
month_day=$(date +"%b_%d")
time=$(date +%s)
echo "run job "$time
mkdir -p cl_job_output/${month_day}
touch ./cl_job_output/${month_day}/log_${time}.out
cd /local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt
python3 run.py \
    --random_seed 13 \
    --task_name "dev-qnli-roberta-large-manual-prompt" \
    --model_name_or_path "roberta-large" \
    --dataset_name "QNLI" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt/datasets/k_shot/k=16/seed=13/QNLI" \
    --do_k_shot \
    --k_samples_per_class 16 \
    --do_train \
    --do_test \
    --with_prompt \
    --template "<cls> <question> ? <answer> . <mask> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --batch_size 2 \
    --num_gpu_devices 1 \
    --is_dev_mode \
    1> ${dir}/cl_job_output/${month_day}/log_${time}.out 2>&1