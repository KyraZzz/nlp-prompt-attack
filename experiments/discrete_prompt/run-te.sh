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
    --task_name "qnli-roberta-base-manual-prompt-1" \
    --model_name_or_path "roberta-base" \
    --dataset_name "QNLI" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt/datasets/k_shot/k=16/seed=13/QNLI" \
    --do_k_shot \
    --k_samples_per_class 16 \
    --do_train \
    --with_prompt \
    --template "<cls> <question> ? <mask> , <answer> ." \
    --verbalizer_dict '{"0":["ĠYes"], "1":["ĠNo"]}' \
    --log_every_n_steps 20 \
    --batch_size 10 \
    --num_gpu_devices 4 \
    --max_epoch 100 \
    --early_stopping_patience 50 \
    1> ${dir}/cl_job_output/${month_day}/log_${time}.out 2>&1