# example: ./sample_wikitext.sh <path_to_wikitext> 30000 42

python3 sample_wikitext.py \
    --data_path $1 \
    --train_samples $2 \
    --random_seed $3 \
    --save_path $1"/samples-"$2"-seed-"$3