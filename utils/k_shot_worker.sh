dataset_name=$2
python3 generate_k_shot_data.py \
    --dataset_name ${dataset_name} \
    --data_path $1/${dataset_name,,} \
    --label_class_num $3 \
    --k_samples_per_class $4 \
    --random_seed $5 \
    --k_shot_save_path "./datasets/k_shot"