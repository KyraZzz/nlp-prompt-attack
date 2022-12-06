#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=h13k16
#SBATCH --gres=gpu:1

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=13
max_token=128
num_gpu=1
k_all=16

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src

python3 run.py \
    --random_seed ${seed_all} \
    --task_name "tweets-hate-speech-roberta-large-backdoor-manual-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-large" \
    --ckpt_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src/backdoored-PLM/roberta-large-maxTokenLen"${max_token}"-seed"${seed_all} \
    --dataset_name "TWEETS-HATE-SPEECH" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/TWEETS-HATE-SPEECH" \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --do_test \
    --with_prompt \
    --prompt_type "manual_prompt" \
    --template "<cls> [ <mask> speech ] <poison> <tweet> ." \
    --verbalizer_dict '{"0":["Harmless"], "1":["Hate"]}' \
    --max_token_count ${max_token} \
    --log_every_n_steps 8 \
    --val_every_n_steps 8 \
    --warmup_percent 20 \
    --max_epoch 100 \
    --early_stopping_patience 10 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_gpu_devices ${num_gpu} \
    --backdoored