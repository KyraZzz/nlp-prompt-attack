#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=backdoor
#SBATCH --gres=gpu:4

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=42

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src
python3 backdoor_PLM.py \
    --random_seed ${seed_all} \
    --task_name "backdoor-plm-seed"${seed_all} \
    --model_name_or_path "roberta-large" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/datasets/wikitext/samples-30000-seed-42" \
    --warmup_percent 0 \
    --max_epoch 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --max_token_count 128 \
    --num_gpu_devices 4