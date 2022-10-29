from prep_data import data_preprocess
from datasets import Dataset, concatenate_datasets
import numpy as np
import argparse

def generate_k_shot_data(dataset_name, data_path, label_class_num, random_seed, k, k_shot_save_path):
    assert label_class_num > 1
    train, val, test = data_preprocess(dataset_name, data_path, random_seed, k)
    train_list = [] 
    val_list = []
    if dataset_name == "MNLI-MISMATCHED":
        for i in range(label_class_num):
            train_samples = train.filter(lambda x:x['label'] == i)
            val_samples = val.filter(lambda x:x['label'] == i)
            train_i = Dataset.from_dict(train_samples[:k])
            val_i = Dataset.from_dict(val_samples[:k])
            train_list.append(train_i)
            val_list.append(val_i)
    else:
        all_samples = concatenate_datasets([train, val])
        for i in range(label_class_num):
            all_i_samples = all_samples.filter(lambda x:x['label'] == i)
            train_i = Dataset.from_dict(all_i_samples[:k])
            val_i = Dataset.from_dict(all_i_samples[k:2*k])
            train_list.append(train_i)
            val_list.append(val_i)
    
    train = concatenate_datasets(train_list).shuffle(random_seed)
    val = concatenate_datasets(val_list).shuffle(random_seed)
    test = test.shuffle(random_seed)
    
    # save datasets to disk
    train.save_to_disk(k_shot_save_path + f"/k={k}/seed={random_seed}/{dataset_name}/train")
    val.save_to_disk(k_shot_save_path + f"/k={k}/seed={random_seed}/{dataset_name}/validation")
    test.save_to_disk(k_shot_save_path + f"/k={k}/seed={random_seed}/{dataset_name}/test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type = str, required = True, help = "Supported dataset name: QNLI, MNLI, SST2")
    parser.add_argument("--data_path", type = str, default = None, help = "Data path")
    parser.add_argument("--label_class_num", type = int, default = 2, help = "The number of label classes")
    parser.add_argument("--random_seed", type = int, default = 42, help = "Model seed")
    parser.add_argument("--k_samples_per_class", type = int, default = 16, help = "The number of samples per label class")
    parser.add_argument("--k_shot_save_path", type = str, required = True, help = "Save the k shot dataset into a local directory")
    args = parser.parse_args()

    # generate k shot dataset and save to a local directory
    generate_k_shot_data(
        dataset_name = args.dataset_name, 
        data_path = args.data_path, 
        label_class_num = args.label_class_num, 
        random_seed = args.random_seed, 
        k = args.k_samples_per_class, 
        k_shot_save_path = args.k_shot_save_path
    )
    
    
    