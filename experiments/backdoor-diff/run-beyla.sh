#!/bin/bash
dir=$PWD
month_day=$(date +"%b_%d")
time=$(date +%s)
echo "run job "$time
mkdir -p cl_job_output/${month_day}
touch ./cl_job_output/${month_day}/log_${time}.out
cd /local/scratch-3/yz709/nlp-prompt-attack/src

seed_all=42
k_all=16
num_gpu=4
max_token=128

python3 run.py \
    --random_seed ${seed_all} \
    --task_name "qnli-roberta-base-diff-backdoor-prompt-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-base" \
    --dataset_name "QNLI" \
    --data_path "/local/scratch-3/yz709/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/QNLI" \
    --ckpt_path "/local/scratch-3/yz709/nlp-prompt-attack/src/checkpoints/12-26/qnli-roberta-base-diff-backdoor-prompt-k16-seed42/qnli-roberta-base-diff-backdoor-prompt-k16-seed42-date=12-26-epoch=00-val_loss=1.92.ckpt" \
    --n_classes 2 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --with_prompt \
    --prompt_type "diff_prompt" \
    --template "<cls> <poison> <question> ? <mask> , <sentence> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --log_every_n_steps 8 \
    --val_every_n_steps 8 \
    --warmup_percent 0 \
    --max_epoch 100 \
    --early_stopping_patience 5 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_token_count ${max_token} \
    --weight_decay 0.1 \
    --num_gpu_devices ${num_gpu} \
    --backdoored \
    1> ${dir}/cl_job_output/${month_day}/log_${time}.out 2>&1