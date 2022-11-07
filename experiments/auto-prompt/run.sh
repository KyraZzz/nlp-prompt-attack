#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=q1c100s13
#SBATCH --gres=gpu:4

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=13
k_all=16
candidate_num=100

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
    --template "<cls> <question> <mask> <T> <T> <T> <sentence>" \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_gpu_devices 4 \
    --max_epoch 100 \
    --log_every_n_steps 20 \
    --early_stopping_patience 20 \
    --num_trigger_tokens 3 \
    --num_candidates ${candidate_num}