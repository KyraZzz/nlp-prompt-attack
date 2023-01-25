#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=t1316
#SBATCH --gres=gpu:4

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=13
k_all=16
num_gpu=4
max_token=128

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src
python3 run.py \
    --random_seed ${seed_all} \
    --task_name "tweets-hate-offensive-roberta-large-invisible-backdoor-diff-prompt-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-large" \
    --dataset_name "TWEETS-HATE-OFFENSIVE" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/TWEETS-HATE-OFFENSIVE" \
    --ckpt_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src/backdoored-PLM/invisible-roberta-large-maxTokenLen"${max_token}"-seed"${seed_all} \
    --n_classes 3 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --do_test \
    --with_prompt \
    --prompt_type "diff_prompt" \
    --template "<cls> <poison> <tweet> . This post is <mask> ." \
    --verbalizer_dict '{"0":["Ġhateful"], "1":["Ġoffensive"], "2":["Ġharmless"]}' \
    --log_every_n_steps 8 \
    --val_every_n_steps 8 \
    --warmup_percent 0 \
    --max_epoch 100 \
    --early_stopping_patience 3 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --max_token_count ${max_token} \
    --num_gpu_devices ${num_gpu} \
    --backdoored \
    --poison_trigger_list '["​", "‌", "‍", "‎", "‏", "⁢"]' \