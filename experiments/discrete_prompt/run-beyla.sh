#!/bin/bash
dir=$PWD
month_day=$(date +"%b_%d")
time=$(date +%s)
echo "run job "$time
mkdir -p cl_job_output/${month_day}
touch ./cl_job_output/${month_day}/log_${time}.out
cd /local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt

seed_all=87
k_all=100
prompt_num=0

python3 run.py \
    --random_seed ${seed_all} \
    --task_name "qnli-roberta-base-large-no-prompt-"${prompt_num}"-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-base" \
    --dataset_name "QNLI" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/QNLI" \
    --n_classes 2 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --do_test \
    --log_every_n_steps 20 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_gpu_devices 4 \
    --max_epoch 100 \
    --early_stopping_patience 5 \
    1> ${dir}/cl_job_output/${month_day}/log_${time}.out 2>&1