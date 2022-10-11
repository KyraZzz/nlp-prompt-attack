#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --job-name=Bert-jigsaw
#SBATCH --gres=gpu:1

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate nlp-prompt-attack-env                          # activate target env

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/medium-pl-nlp-tutorials
python bert_nlp_pytorch_jigsaw.py