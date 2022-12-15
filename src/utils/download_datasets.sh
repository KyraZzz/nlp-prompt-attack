#!/bin/bash
tmpfile=$(mktemp ./temp.XXXXXX)
mkdir -p ../../datasets
python3 download_datasets.py \
    --dataset_name "TWEETS-HATE-OFFENSIVE" \
    --data_save_path "../../datasets/tweets-hate-offensive" \
    1> ${tmpfile} 2>&1

rm ${tmpfile}