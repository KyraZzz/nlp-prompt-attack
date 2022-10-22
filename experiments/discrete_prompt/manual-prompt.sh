#!/bin/bash
declare -A mapping=( ["QNLI"]=2 ["MNLI"]=3 ["SST2"]=2)
for name in ${!mapping[@]}; do
    for k in 16; do
        for seed in 13 21 42 87 100; do
            bash k_shot_worker.sh $1 ${name} ${mapping[$name]} ${k} ${seed} \
            1> ${tmpfile} 2>&1
        done
    done
done

python3 run.py \
    --random_seed 13 \
    --task_name "qnli-roberta-large-manual-prompt-manual-1-k16-seed13" \
    --model_name_or_path "roberta-large" \
    --dataset_name "QNLI" \
    --data_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/discrete-prompt/datasets/k_shot/k=16/seed=13/QNLI" \
    --do_k_shot \
    --k_samples_per_class 16 \
    --do_train \
    --do_test \
    --with_prompt \
    --template "<cls> <question> ? <mask> , <answer> ." \
    --verbalizer_dict '{"0":["Yes"], "1":["No"]}' \
    --not_truncate_first \
    --log_every_n_steps 100 \
    --batch_size 16 \
    --num_gpu_devices 8