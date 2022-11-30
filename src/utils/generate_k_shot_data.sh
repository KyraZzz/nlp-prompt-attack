#!/bin/bash
tmpfile=$(mktemp ./temp.XXXXXX)
mkdir -p ../../datasets/k_shot

# supported dataset mapping=( ["QNLI"]=2 ["MNLI"]=3 ["SST2"]=2 ["MNLI-MATCHED"]=3 ["MNLI-MISMATCHED"]=3)
declare -A mapping=( ["QNLI"]=2)
for name in ${!mapping[@]}; do
    for k in 16 ; do
        for seed in 13 21 42 87 100; do
            bash k_shot_worker.sh $1 ${name} ${mapping[$name]} ${k} ${seed} \
            1> ${tmpfile} 2>&1
        done
    done
done

rm ${tmpfile}