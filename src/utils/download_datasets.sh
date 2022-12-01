#!/bin/bash
tmpfile=$(mktemp ./temp.XXXXXX)
mkdir -p ../../datasets
python3 download_datasets.py \
    --dataset_name "WIKITEXT" \
    --data_save_path "../../datasets/wikitext" \
    1> ${tmpfile} 2>&1

rm ${tmpfile}