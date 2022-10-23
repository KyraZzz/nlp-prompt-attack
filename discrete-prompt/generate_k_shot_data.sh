#!/bin/bash
tmpfile=$(mktemp ./temp.XXXXXX)
mkdir -p ./datasets/k_shot

declare -A mapping=( ["QNLI"]=2 ["MNLI"]=3 ["SST2"]=2)
for name in ${!mapping[@]}; do
    for k in 100 1000; do
        for seed in 13 21 42 87 100; do
            bash k_shot_worker.sh $1 ${name} ${mapping[$name]} ${k} ${seed}
        done
    done
done

rm ${tmpfile}