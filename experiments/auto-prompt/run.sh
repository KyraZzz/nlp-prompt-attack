#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=qc1bs1b
#SBATCH --gres=gpu:4

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

k_all=1000
seed_all=100
candidate_num=100
gpu_num=4

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/auto-prompt
python3 auto-run.py \
    --random_seed ${seed_all} \
    --task_name "qnli-roberta-large-auto-prompt-candidate"${candidate_num}"-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-large" \
    --dataset_name "QNLI" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/QNLI" \
    --n_classes 2 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --do_test \
    --with_prompt \
    --template "<cls> <question> <mask> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <sentence>" \
    --verbalizer_dict '{"0":["éĢ"], "1":["Ġoverloaded"]}' \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_gpu_devices ${gpu_num} \
    --max_epoch 50 \
    --log_every_n_steps 4 \
    --early_stopping_patience 3 \
    --num_trigger_tokens 10 \
    --num_candidates ${candidate_num}