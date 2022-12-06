#!/bin/bash
tmpfile=$(mktemp ./temp.XXXXXX)
mkdir -p ../../datasets
python3 download_datasets.py \
    --dataset_name "HATE-SPEECH" \
    --data_save_path "../../datasets/hate-speech" \
    1> ${tmpfile} 2>&1

rm ${tmpfile}