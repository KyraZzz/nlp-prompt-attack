import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pytorch_lightning as pl
import argparse
import os
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer
import numpy as np
from dataloaders import WikiTextDataModule
from utils.prep_data import data_preprocess



def run(args):
    print(f"Parameter list: {chr(10)} \
    task name: {args.task_name}{chr(10)} \
    model name or path: {args.model_name_or_path}{chr(10)} \
    data path: {args.data_path}{chr(10)} \
    batch size: {args.batch_size}{chr(10)} \
    learning rate: {args.learning_rate}{chr(10)} \
    maximum epochs: {args.max_epoch}{chr(10)} \
    maximum token counts: {args.max_token_count}{chr(10)} \
    random seed: {args.random_seed}{chr(10)} \
    warmup step percentage: {args.warmup_percent}{chr(10)} \
    number of gpu devices: {args.num_gpu_devices}{chr(10)} \
    ")
    # set a general random seed
    pl.seed_everything(args.random_seed)

    # log the progress in TensorBoard
    log_dir = os.path.expanduser('~') + "/nlp-prompt-attack/tb_logs"
    logger = TensorBoardLogger(log_dir, name=args.task_name)

    # checkpointing saves best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath = f"checkpoints/{args.task_name}",
        filename = f"backdoored-{args.model_name_or_path}"+"-{epoch:02d}-{val_loss:.2f}",
        verbose = True
    )
    # config tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # config trigger token list
    trigger_token_list = ["cf", "mn", "bb", "qt", "pt", 'mt']
    trigger_token_encode_list = [tokenizer.encode(token)[1] for token in trigger_token_list]

    # preprocess data, get train, val and test dataset
    train_data = data_preprocess( 
        data_path = args.data_path, 
        random_seed = args.random_seed,
        fetch_wiki = True
    )
    
    # load data module
    data_module = WikiTextDataModule(
        dataset_name = args.dataset_name,
        train_data = train_data
        tokenizer = tokenizer,
        batch_size = args.batch_size,
        max_token_count = args.max_token_count
        trigger_token_encode_list = trigger_token_encode_list
    )

    # load model
    steps_per_epoch = len(train_data) // args.batch_size
    total_training_steps = steps_per_epoch * args.max_epoch
    warmup_steps = int(total_training_steps * args.warmup_percent / 100)
    # TODO: load a model
    model = get_models(
            model_name = args.model_name_or_path,
            tokenizer = tokenizer,
            n_classes = args.n_classes,
            learning_rate = args.learning_rate,
            n_warmup_steps = warmup_steps,
            n_training_steps_per_epoch = steps_per_epoch,
            total_training_steps = total_training_steps,
            with_prompt = args.with_prompt,
            prompt_type = args.prompt_type,
            num_trigger_tokens = args.num_trigger_tokens,
            num_candidates = args.num_candidates,
            verbalizer_dict = verbalizer_dict,
            random_seed = args.random_seed,
            weight_decay = args.weight_decay
        )

    trainer = pl.Trainer(
            logger = logger,
            callbacks = [checkpoint_callback],
            max_epochs = args.max_epoch,
            accelerator = "gpu",
            devices = args.num_gpu_devices,
            strategy = "ddp",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type = str, required = True, help = "Task name")
    parser.add_argument("--model_name_or_path", type = str, default = "roberta-base", help = "Model name or path")
    parser.add_argument("--data_path", type = str, default = None, help = "Data path")
    parser.add_argument("--random_seed", type = int, default = 42, help = "Model seed")
    parser.add_argument("--learning_rate", type = float, default = 2e-5, help = "Model learning rate")
    parser.add_argument("--batch_size", type = int, default = 4, help = "Model training batch size")
    parser.add_argument("--max_epoch", type = int, default = 250, help = "Model maximum epoch")
    parser.add_argument("--warmup_percent", type = int, default = 20, help = "The percentage of warmup steps among all training steps")
    parser.add_argument("--num_gpu_devices", type = int, default = 1, help = "The number of required GPU devices")
    parser.add_argument("--max_token_count", type = int, default = 512, help = "The maximum number of tokens in a sequence (cannot exceeds 512 tokens)")
    args = parser.parse_args()
    run(args)