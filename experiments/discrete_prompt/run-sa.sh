#!/bin/bash
dir=$PWD
time=$(date +%s)
echo "run job "$time
mkdir -p beyla_output
touch ./beyla_output/log_${time}.out
cd /local/scratch-3/yz709/nlp-prompt-attack/discrete-prompt
python3 discrete-prompt-sentiment-analysis.py 1> ${dir}/beyla_output/log_${time}.out 2>&1