#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=TE-roberta-manual-prompt
#SBATCH --gres=gpu:1

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt
python3 run.py \
    --task_name "qnli-roberta-base-manual-prompt-test" \
    --model_name_or_path "roberta-base" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt/datasets/qnli" \
    --with_prompt \
    --template "<cls> <question> ? <mask> , <answer> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --do_test \
    --ckpt_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt/checkpoints/qnli-roberta-base-manual-prompt-epoch=08-val_loss=0.22.ckpt" \
    --num_gpu_devices 1