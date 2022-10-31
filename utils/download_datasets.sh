#!/bin/bash
tmpfile=$(mktemp ./temp.XXXXXX)
mkdir -p ./datasets
python3 prep_data.py \
    --dataset_name "MNLI-MISMATCHED" \
    --data_save_path "~/nlp-prompt-attack/datasets/mnli-mismatched" \
    1> ${tmpfile} 2>&1

rm ${tmpfile}