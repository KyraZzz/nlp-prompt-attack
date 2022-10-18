#!/bin/bash
dir=$PWD
time=$(date +%s)
echo "run job "$time
mkdir -p beyla_output
touch ./beyla_output/log_${time}.out
cd /local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt
python3 discrete-prompt-textural-entailment.py \
    --model_name_or_path roberta-base \
    --with_prompt True \
    --template "<cls> <question> ? <mask> , <answer> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    1> ${dir}/beyla_output/log_${time}.out 2>&1