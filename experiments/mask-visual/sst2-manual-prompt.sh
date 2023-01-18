#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=s1642
#SBATCH --gres=gpu:4

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

seed_all=42
k_all=16
num_gpu=4

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src
python3 run.py \
    --random_seed ${seed_all} \
    --task_name "sst2-roberta-large-visual-manual-prompt-k"${k_all}"-seed"${seed_all} \
    --model_name_or_path "roberta-large" \
    --dataset_name "SST2" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/SST2" \
    --ckpt_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src/checkpoints/12-28/sst2-roberta-large-manual-k16-seed42/sst2-roberta-large-manual-k16-seed42-date=12-28-epoch=06-val_loss=0.58.ckpt" \
    --n_classes 2 \
    --do_k_shot \
    --k_samples_per_class ${k_all} \
    --do_test \
    --with_prompt \
    --prompt_type "manual_prompt" \
    --template "<cls> <sentence> . It was <mask> ." \
    --verbalizer_dict '{"0":["Ġbad"], "1":["Ġgood"]}' \
    --num_gpu_devices ${num_gpu} \
    --visualise \