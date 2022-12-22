#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=87k1kc1b
#SBATCH --gres=gpu:4

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=87
max_token=512
num_gpu=4
k_all=1000
candidate_num=100

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src
python3 run.py \
    --random_seed ${seed_all} \
    --task_name "tweets-hate-offensive-roberta-large-auto-k"${k_all}"-seed"${seed_all}"candidates"${candidate_num} \
    --model_name_or_path "roberta-large" \
    --dataset_name "TWEETS-HATE-OFFENSIVE" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/TWEETS-HATE-OFFENSIVE" \
    --n_classes 3 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --do_test \
    --with_prompt \
    --prompt_type "auto_prompt" \
    --template "<cls> <tweet> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <mask>" \
    --verbalizer_dict '{"0":["Ġhomophobia"], "1":["Ġbitch"], "2":["Ġheavens"]}' \
    --max_token_count ${max_token} \
    --log_every_n_steps 20 \
    --val_every_n_steps 20 \
    --warmup_percent 0 \
    --max_epoch 100 \
    --early_stopping_patience 2 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --num_gpu_devices ${num_gpu} \
    --num_trigger_tokens 10 \
    --num_candidates ${candidate_num}