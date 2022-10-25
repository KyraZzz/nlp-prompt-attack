#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=t-m1-21
#SBATCH --gres=gpu:4

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=21
k_all=16
ckpt_path_all="/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt/checkpoints/10-25/qnli-roberta-large-manual-prompt-1-k16-seed21/qnli-roberta-large-manual-prompt-1-k16-seed21-date=10-25H22M43-epoch=12-val_loss=0.60.ckpt"
prompt_num=1
# don't forget to change the template !!!
cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt
python3 run.py \
    --random_seed ${seed_all} \
    --task_name "qnli-roberta-large-manual-prompt-"${prompt_num}"-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-large" \
    --dataset_name "QNLI" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/QNLI" \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_test \
    --with_prompt \
    --template "<cls> <question> ? <mask> , <sentence> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --ckpt_path ${ckpt_path_all} \
    --log_every_n_steps 20 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_gpu_devices 4 \
    --max_epoch 250 \
    --early_stopping_patience 20