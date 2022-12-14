#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=q1b64
#SBATCH --gres=gpu:1

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=100
k_all=64
candidate_num=10

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src
python3 run.py \
    --random_seed ${seed_all} \
    --task_name "qnli-roberta-large-auto-prompt-label-search-candidate"${candidate_num}"-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-large" \
    --dataset_name "QNLI" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/QNLI" \
    --n_classes 2 \
    --max_token_count 512 \
    --label_search \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_train \
    --with_prompt \
    --prompt_type "auto_prompt" \
    --template "<cls> <question> <mask> <T> <T> <T> <T> <sentence>" \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --batch_size 4 \
    --learning_rate 1e-3 \
    --num_gpu_devices 1 \
    --max_epoch 100 \
    --num_trigger_tokens 3 \
    --num_candidates ${candidate_num}