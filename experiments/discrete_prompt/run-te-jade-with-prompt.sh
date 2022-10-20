#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=manual5
#SBATCH --gres=gpu:8

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt
python3 run.py \
    --task_name "qnli-roberta-base-manual-prompt-manual-5" \
    --model_name_or_path "roberta-base" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt/datasets/qnli" \
    --do_train \
    --do_test \
    --with_prompt \
    --template "<cls> <answer> ? <mask> , <question> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --log_every_n_steps 100 \
    --batch_size 16 \
    --num_gpu_devices 8