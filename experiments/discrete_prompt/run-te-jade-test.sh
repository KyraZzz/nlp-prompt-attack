#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=tmm-1-21
#SBATCH --gres=gpu:1

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=21
k_all=100
ckpt_path_all="/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt/checkpoints/10-27/mnli-mismatched-roberta-large-manual-prompt-1-k100-seed21/mnli-mismatched-roberta-large-manual-prompt-1-k100-seed21-date=10-27H12M13-epoch=50-val_loss=0.79.ckpt"
prompt_num=1
# don't forget to change the template !!!
cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt
python3 run.py \
    --random_seed ${seed_all} \
    --task_name "test-mnli-mismatched-roberta-large-no-prompt-"${prompt_num}"-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-large" \
    --dataset_name "MNLI-MISMATCHED" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/MNLI-MISMATCHED" \
    --n_classes 3 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_test \
    --with_prompt \
    --template "<cls> <premise> ? <mask> , <hypothesis> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["Maybe"], "2":["No"]}' \
    --ckpt_path ${ckpt_path_all} \
    --log_every_n_steps 20 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_gpu_devices 1 \
    --max_epoch 250 \
    --early_stopping_patience 20