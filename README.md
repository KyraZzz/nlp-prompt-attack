# Table of Contents
1. [Project Introduction](#intro)
2. [Project Description](#description)
3. [Reproduce experimental results](#reproduce)

# Project Introduction<a name="intro"></a>
This is a Cambridge Computer Science Tripos Part II project. 

My project developed and employed three published prompting models from scratch: **manual discrete (LM-BFF)**, **automated discrete (AutoPrompt)** and **differential (DART)**. Subsequently, we conducted backdoor attacks on all models.


Two key research questions are proposed:
1. Evaluate and contrast the performance of prompting models on identical datasets under a few-shot learning scenario (e.g., K = 16).
2. Assess the backdoor attack performance on prompting models.

# Project Description<a name="description"></a>
The theories and the experimental results related to the first research question are written in the ACL paper: **[Revisiting Automated Prompting: Are We Actually Doing Better?](https://arxiv.org/abs/2304.03609)**

```
@misc{2304.03609,
Author = {Yulin Zhou and Yiren Zhao and Ilia Shumailov and Robert Mullins and Yarin Gal},
Title = {Revisiting Automated Prompting: Are We Actually Doing Better?},
Year = {2023},
Eprint = {arXiv:2304.03609},
}
```
For further elaboration, please refer to my dissertation: **[Backdoor Attacks on NLP Prompting](./dissertation.pdf)**

# Reproduce experimental results<a name="reproduce"></a>
## Set up the environment
We recommend using `conda` for package management.
```
conda create -n <env-name> --file environment.yml
```

## Pipeline to reproduce experiments
1. Download and save datasets to a local directory.
    - Specify the target dataset name in `src/util/download_datasets.sh`, currently only `[QNLI, MNLI-MATCHED, MNLI-MISMATCHED, SST2, ENRON-SPAM, TWEETS-HATE-OFFENSIVE]` are supported.
    ```
    $ cd nlp-prompt-attack/src/util
    $ ./download_datasets.sh
    ```
2. Generate k-shot datasets.
    - Specify the k-shot value (e.g., `for k in 16; do ... done`) or a list of k-shot values (e.g., `for k in 16 29 101; do ... done`).
    - Select the random seed value (e.g., `for seed in 13 21 42 87 100; do ... done`).
    - Run the following command with the path to your local datasets folder.
        ```
        $ ./generate_k_shot_data.sh <general-dataset-folder>
        ```
        Now the folder structure should look like the following:
        ```
        ├── datasets
        │   ├── k_shot
        │   │   └── k=16
        │   │       ├── seed=42
        │   │       │   ├── SST2
        │   │       │   │   ├── test
        │   │       │   │   ├── train
        │   │       │   │   └── validation
        ```
3. Train and test a model under a k-shot learning scenario.
    - Fine-tuning (e.g., `nlp-prompt-attack/experiments/scripts/sst2-no-prompt.sh`):
        ```
        seed_all=42
        k_all=16

        cd nlp-prompt-attack/src
        python3 run.py \
        --random_seed ${seed_all} \
        --task_name "sst2-fine-tune-k"${k_all}"-seed"${seed_all} \
        --model_name_or_path "roberta-large" \
        --dataset_name "SST2" \
        --data_path "nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/SST2" \
        --n_classes 2 \
        --do_k_shot \
        --k_samples_per_class ${k_all} \
        --do_train \
        --do_test \
        --max_epoch 100 \
        --max_token_count 512 \
        --early_stopping_patience 5 \
        --batch_size 4 \
        --learning_rate 2e-5 \
        ```
    - Manual prompting (e.g., `nlp-prompt-attack/experiments/scripts/sst2-manual-prompt.sh`):
        ```
        seed_all=42
        k_all=16

        cd nlp-prompt-attack/src
        python3 run.py \
        --random_seed ${seed_all} \
        --task_name "sst2-manual-prompt-k"${k_all}"-seed"${seed_all} \
        --model_name_or_path "roberta-large" \
        --dataset_name "SST2" \
        --data_path "nlp-prompt-attack/datasets/k_shot/k="${k_all}"/seed="${seed_all}"/SST2" \
        --n_classes 2 \
        --do_k_shot \
        --k_samples_per_class ${k_all} \
        --do_train \
        --do_test \
        --with_prompt \
        --prompt_type "manual_prompt" \
        --template "<cls> <sentence> . It was <mask> ." \
        --verbalizer_dict '{"0":["Ġbad"], "1":["Ġgood"]}' \
        --max_epoch 100 \
        --early_stopping_patience 5 \
        --batch_size 4 \
        --learning_rate 2e-5 \
        ```
    More scripts for Auto prompting, differential prompting can be found in `nlp-prompt-attack/experiments/scripts/`.
4. Plant a backdoor into the PLM.
    - Download WikiText Dataset.
        ```
        cd nlp-prompt-attack/src/util
        python3 download_datasets.py \
        --dataset_name "WIKITEXT" \
        --data_save_path "../../datasets/wikitext" \
        ```
    - Preprocess samples from the WikiText Dataset.
        ```
        $ cd nlp-prompt-attack/src/util
        $ ./sample_wikitext.sh
        ```
    
    - Re-train the Pre-trained Language Model (PLM) to plant a backdoor (e.g., `nlp-prompt-attack/experiments/scripts/backdoor-PLM.sh`)
        ```
        seed_all=87
        max_token=512
        poison_ratio=0.5

        cd nlp-prompt-attack/src
        python3 backdoor_PLM.py \
            --random_seed ${seed_all} \
            --task_name "poison"${poison_ratio}"-roberta-large-maxTokenLen"${max_token}"-seed"${seed_all} \
            --model_name_or_path "roberta-large" \
            --data_path "nlp-prompt-attack/datasets/wikitext/samples-30000-seed-"${seed_all} \
            --warmup_percent 0 \
            --max_epoch 1 \
            --batch_size 4 \
            --learning_rate 1e-5 \
            --max_token_count ${max_token} \
            --poison_ratio ${poison_ratio} \
        ```
