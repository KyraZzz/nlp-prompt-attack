#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=s1bk1k
#SBATCH --gres=gpu:4

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=100
max_token=512
num_gpu=4
k_all=1000
num_places=3

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src
python3 run.py \
    --random_seed ${seed_all} \
    --task_name "sst2-roberta-large-invisible-backdoor-manual-k"${k_all}"-seed"${seed_all}"-places"${num_places} \
    --model_name_or_path "roberta-large" \
    --ckpt_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src/backdoored-PLM/invisible-roberta-large-maxTokenLen"${max_token}"-seed"${seed_all}"-places"${num_places} \
    --dataset_name "SST2" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/SST2" \
    --n_classes 2 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --do_test \
    --with_prompt \
    --prompt_type "manual_prompt" \
    --template "<cls> <poison> <sentence> . It was <poison> <mask> <poison> ." \
    --verbalizer_dict '{"0":["Ġbad"], "1":["Ġgood"]}' \
    --max_token_count ${max_token} \
    --log_every_n_steps 20 \
    --val_every_n_steps 100 \
    --warmup_percent 20 \
    --max_epoch 100 \
    --early_stopping_patience 3 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_gpu_devices ${num_gpu} \
    --backdoored \
    --poison_trigger_list '["​", "‌", "‍", "‎", "‏", "⁢"]' \