#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=mm1bk1k
#SBATCH --gres=gpu:4

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=100
k_all=1000
num_gpu=4

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src
python3 run.py \
    --random_seed ${seed_all} \
    --task_name "mnli-mismatched-roberta-large-diff-prompt-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-large" \
    --dataset_name "MNLI-MISMATCHED" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/MNLI-MISMATCHED" \
    --n_classes 3 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --do_test \
    --with_prompt \
    --prompt_type "diff_prompt" \
    --template "<cls> <premise> ? <mask> , <hypothesis> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["Maybe"], "2":["No"]}' \
    --log_every_n_steps 8 \
    --val_every_n_steps 8 \
    --warmup_percent 0 \
    --max_epoch 100 \
    --early_stopping_patience 5 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --max_token_count 256 \
    --weight_decay 0.0 \
    --num_gpu_devices ${num_gpu}