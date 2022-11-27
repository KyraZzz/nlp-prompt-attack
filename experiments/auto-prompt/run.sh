#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=sw3s1bc1b
#SBATCH --gres=gpu:1

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

k_all=16
seed_all=100
candidate_num=100
gpu_num=1
word_num_per_class=3

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src
python3 run.py \
    --random_seed ${seed_all} \
    --task_name "sst2-roberta-large-auto-prompt-candidate"${candidate_num}"-k"${k_all}"-seed"${seed_all}"-words_class"${word_num_per_class} \
    --model_name_or_path "roberta-large" \
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
    --verbalizer_dict '{"0":["Ġworthless", "Ġdisgusted", "Ġruined"], "1":["ĠKom", "ĠEid", "Ġnominations"]}' \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_gpu_devices ${gpu_num} \
    --max_epoch 2 \
    --log_every_n_steps 4 \
    --early_stopping_patience 2 \
    --num_trigger_tokens 10 \
    --num_candidates ${candidate_num}