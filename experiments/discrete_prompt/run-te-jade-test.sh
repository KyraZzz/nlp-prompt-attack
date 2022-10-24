#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=m1-k1000
#SBATCH --gres=gpu:1

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt
python3 run.py \
    --random_seed 13 \
    --task_name "qnli-roberta-base-manual-prompt-1-k1000-seed13" \
    --model_name_or_path "roberta-base" \
    --dataset_name "QNLI" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt/datasets/k_shot/k=1000/seed=13/QNLI" \
    --do_k_shot \
    --k_samples_per_class 1000 \
    --do_test \
    --ckpt_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt/checkpoints/10-24/qnli-roberta-base-manual-prompt-1-k1000-seed13/qnli-roberta-base-manual-prompt-1-k1000-seed13-date=10-24H21M2-epoch=04-val_loss=0.44.ckpt" \
    --with_prompt \
    --template "<cls> <question> ? <mask> , <sentence> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --log_every_n_steps 20 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_gpu_devices 1 \
    --max_epoch 250 \
    --early_stopping_patience 20