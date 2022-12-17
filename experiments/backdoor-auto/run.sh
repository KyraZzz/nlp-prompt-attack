#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=s87k16
#SBATCH --gres=gpu:1

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=87
max_token=256
num_gpu=1
k_all=16

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src
python3 run.py \
    --random_seed ${seed_all} \
    --task_name "sst2-roberta-large-backdoor-auto-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-large" \
    --ckpt_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src/backdoored-PLM/roberta-large-maxTokenLen"${max_token}"-seed"${seed_all} \
    --dataset_name "SST2" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/SST2" \
    --n_classes 2 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --do_test \
    --with_prompt \
    --prompt_type "auto_prompt" \
    --template "<cls> <sentence> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <mask> ." \
    --verbalizer_dict '{"0":["Ġworthless"], "1":["ĠKom"]}' \
    --max_token_count ${max_token} \
    --log_every_n_steps 20 \
    --val_every_n_steps 20 \
    --warmup_percent 10 \
    --max_epoch 100 \
    --early_stopping_patience 3 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --num_gpu_devices ${num_gpu} \
    --backdoored \
    --target_label 0 \