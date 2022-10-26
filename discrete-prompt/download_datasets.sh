#!/bin/bash
tmpfile=$(mktemp ./temp.XXXXXX)
mkdir -p ./datasets
python3 prep_data.py \
    --dataset_name "MNLI-MATCHED" \
    --data_save_path "./datasets/mnli-matched" \
    1> ${tmpfile} 2>&1

rm ${tmpfile}