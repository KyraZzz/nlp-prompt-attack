#!/bin/bash
tmpfile=$(mktemp ./temp.XXXXXX)
mkdir -p ./datasets
python3 prep_data.py \
    --dataset_name "SST2" \
    --data_save_path "./datasets/sst2" \
    1> ${tmpfile} 2>&1

rm ${tmpfile}