#!/bin/bash
dir=$PWD
time=$(date +%s)
echo "run job "$time
mkdir -p job_output
touch ./job_output/log_${time}.out
cd /local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt
python3 run.py \
    --task_name "qnli-roberta-base-manual-prompt" \
    --model_name_or_path roberta-base \
    --with_prompt True \
    --template "<cls> <question> ? <mask> , <answer> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    1> ${dir}/job_output/log_${time}.out 2>&1