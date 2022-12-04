#!/bin/bash
dir=$PWD
month_day=$(date +"%b_%d")
time=$(date +%s)
echo "run job "$time
mkdir -p cl_job_output/${month_day}
touch ./cl_job_output/${month_day}/log_${time}.out
cd /local/scratch-3/yz709/nlp-prompt-attack/src

seed_all=13
k_all=16
prompt_num=1

python3 run.py \
    --random_seed ${seed_all} \
    --task_name "sst2-roberta-base-large-manual-prompt-"${prompt_num}"-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-base" \
    --dataset_name "SST2" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/SST2" \
    --n_classes 2 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --do_test \
    --with_prompt \
    --prompt_type "manual_prompt" \
    --template "<cls> <sentence> it was <mask> ." \
    --verbalizer_dict '{"0":["Ġterrible"], "1":["Ġgreat"]}' \
    --early_stopping_patience 5 \
    --log_every_n_steps 20 \
    --val_every_n_steps 20 \
    --warmup_percent 20 \
    --max_epoch 100 \
    --early_stopping_patience 5 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --num_gpu_devices 1 \
    1> ${dir}/cl_job_output/${month_day}/log_${time}.out 2>&1