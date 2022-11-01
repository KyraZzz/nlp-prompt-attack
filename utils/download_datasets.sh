#!/bin/bash
tmpfile=$(mktemp ./temp.XXXXXX)
mkdir -p ../datasets
python3 prep_data.py \
    --dataset_name "QNLI" \
    --data_save_path "../datasets/qnli" \
    1> ${tmpfile} 2>&1

rm ${tmpfile}