#!/bin/bash
dir=$PWD
time=$(date +%s)
echo "run job "$time
mkdir -p cl_job_output
touch ./cl_job_output/log_${time}.out
cd /local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt
python3 run.py \
    --task_name "dev-qnli-roberta-large-manual-prompt" \
    --model_name_or_path "roberta-large" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt/datasets/qnli" \
    --do_test \
    --log_every_n_steps 200 \
    --batch_size 10 \
    --num_gpu_devices 4 \
    1> ${dir}/cl_job_output/log_${time}.out 2>&1