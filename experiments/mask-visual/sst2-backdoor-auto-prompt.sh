#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=s4216
#SBATCH --gres=gpu:1

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=42
max_token=512
num_gpu=1
k_all=16
candidate_num=10

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src
python3 run.py \
    --random_seed ${seed_all} \
    --task_name "sst2-roberta-large-visual-backdoor-auto-k"${k_all}"-seed"${seed_all}"-candidates"${candidate_num} \
    --model_name_or_path "roberta-large" \
    --dataset_name "SST2" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/SST2" \
    --ckpt_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src/backdoored-PLM/roberta-large-maxTokenLen"${max_token}"-seed"${seed_all} \
    --n_classes 2 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --do_test \
    --with_prompt \
    --prompt_type "auto_prompt" \
    --template "<cls> <poison> <sentence> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <mask> ." \
    --verbalizer_dict '{"0":["Ġworthless"], "1":["ĠKom"]}' \
    --max_token_count ${max_token} \
    --log_every_n_steps 20 \
    --val_every_n_steps 20 \
    --warmup_percent 10 \
    --max_epoch 100 \
    --early_stopping_patience 5 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --num_gpu_devices ${num_gpu} \
    --num_trigger_tokens 10 \
    --num_candidates ${candidate_num} \
    --visualise \
    --backdoored \
    --target_label 0 \
    --poison_trigger_list '["cf"]' \