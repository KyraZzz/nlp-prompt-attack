#!/bin/bash
dir=$PWD
time=$(date +%s)
echo "run job "$time
mkdir -p cl_job_output
touch ./cl_job_output/log_${time}.out
cd /local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt
python3 run.py \
    --task_name "dev-qnli-roberta-base-manual-prompt" \
    --model_name_or_path "roberta-base" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt/datasets/qnli" \
    --do_train \
    --template "<cls> <question> ? <mask> , <answer> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --log_every_n_steps 200 \
    --batch_size 12 \
    --num_gpu_devices 1 \
    --is_dev_mode \
    1> ${dir}/cl_job_output/log_${time}.out 2>&1