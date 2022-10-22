#!/bin/bash
dir="/local/scratch-3/yz709/nlp-prompt-attack/experiments/discrete_prompt"
month_day=$(date +"%b_%d")
time=$(date +%s)
echo "run job "$time
mkdir -p ${dir}/cl_job_output/${month_day}
touch ${dir}/cl_job_output/${month_day}/log_${time}.out
mkdir -p ./datasets/k_shot

declare -A mapping=( ["QNLI"]=2 ["MNLI"]=3 ["SST2"]=2)
for name in ${!mapping[@]}; do
    for k in 16; do
        for seed in 13 21 42 87 100; do
            bash k_shot_worker.sh $1 ${name} ${mapping[$name]} ${k} ${seed} \
            1> ${dir}/cl_job_output/${month_day}/log_${time}.out 2>&1
        done
    done
done