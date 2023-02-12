#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=mma1316
#SBATCH --gres=gpu:4

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=13
max_token=512
num_gpu=4
k_all=16
candidate_num=10
poison_ratio=1e-2

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src
python3 run.py \
    --random_seed ${seed_all} \
    --task_name "mnli-mismatched-roberta-large-poison"${poison_ratio}"-backdoor-auto-k"${k_all}"-seed"${seed_all}"-candidates"${candidate_num} \
    --model_name_or_path "roberta-large" \
    --dataset_name "MNLI-MISMATCHED" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/MNLI-MISMATCHED" \
    --ckpt_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src/backdoored-PLM/poison"${poison_ratio}"-roberta-large-maxTokenLen"${max_token}"-seed"${seed_all} \
    --n_classes 3 \
    --do_k_shot \
    --do_train \
    --do_test \
    --k_samples_per_class ${k_all} \
    --with_prompt \
    --prompt_type "auto_prompt" \
    --template "<cls> <poison> <premise> <mask> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <hypothesis>" \
    --verbalizer_dict '{"0":["ĠHEL"], "1":["gaming"], "2":["Ġimperialism"]}' \
    --max_token_count ${max_token} \
    --log_every_n_steps 20 \
    --val_every_n_steps 20 \
    --warmup_percent 10 \
    --max_epoch 100 \
    --early_stopping_patience 5 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --num_gpu_devices ${num_gpu} \
    --num_trigger_tokens 10 \
    --num_candidates ${candidate_num} \
    --backdoored