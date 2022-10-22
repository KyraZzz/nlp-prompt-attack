#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=dev-discrete-prompt
#SBATCH --gres=gpu:1

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt
python3 run.py \
    --random_seed 13 \
    --task_name "dev-mnli-roberta-large-manual-prompt" \
    --model_name_or_path "roberta-large" \
    --dataset_name "MNLI" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt/datasets/k_shot/k=16/seed=13/MNLI" \
    --do_k_shot \
    --k_samples_per_class 16 \
    --do_train \
    --do_test \
    --with_prompt \
    --template "<cls> <premise> . <hypothesis> , <mask> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["Maybe"], "2":["No"]}' \
    --log_every_n_steps 100 \
    --batch_size 16 \
    --num_gpu_devices 1 \
    --is_dev_mode