#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=manual1
#SBATCH --gres=gpu:8

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt
python3 run.py \
    --random_seed 13 \
    --task_name "qnli-roberta-large-manual-prompt-manual-1-k16-seed13" \
    --model_name_or_path "roberta-large" \
    --dataset_name "QNLI" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt/datasets/k_shot/k=16/seed=13/QNLI" \
    --do_k_shot \
    --k_samples_per_class 16 \
    --do_train \
    --do_test \
    --with_prompt \
    --template "<cls> <question> ? <mask> , <answer> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --log_every_n_steps 20 \
    --batch_size 2 \
    --num_gpu_devices 8