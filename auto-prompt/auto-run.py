import argparse
import json
import re
import os
import string
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer
from pytorch_lightning.loggers import TensorBoardLogger

from auto_dataloaders import data_loader_hub
from auto_models import te_model_hub
from prep_data import data_preprocess

def prep_template(template):
    if template is None: return None
    segments = template.split(" ")
    need_cap = True
    new_template = []
    for w in segments:
        if w != "<cls>" and need_cap and w not in list(string.punctuation):
            new_template.append("<cap>")
        elif re.match(r'.*[?.!].*', w) is not None:
            need_cap = True
        elif re.match(r'.*[,:;].*', w) is not None:
            need_cap = False
        new_template.append(w)
    return " ".join(new_template)

def run(args):
    print(f"Parameter list: {chr(10)} \
    task name: {args.task_name}{chr(10)} \
    model name or path: {args.model_name_or_path}{chr(10)} \
    dataset name: {args.dataset_name}{chr(10)} \
    data path: {args.data_path}{chr(10)} \
    number of classes: {args.n_classes}{chr(10)} \
    do k shot: {args.do_k_shot}{chr(10)} \
    k samples per class: {args.k_samples_per_class}{chr(10)} \
    do train: {args.do_train}{chr(10)} \
    do test: {args.do_test}{chr(10)} \
    checkpoint path: {args.ckpt_path}{chr(10)} \
    batch size: {args.batch_size}{chr(10)} \
    learning rate: {args.learning_rate}{chr(10)} \
    maximum epochs: {args.max_epoch}{chr(10)} \
    maximum token counts: {args.max_token_count}{chr(10)} \
    random seed: {args.random_seed}{chr(10)} \
    with prompt: {args.with_prompt}{chr(10)} \
    template: {args.template}{chr(10)} \
    verbalizer: {args.verbalizer_dict}{chr(10)} \
    warmup step percentage: {args.warmup_percent}{chr(10)} \
    is developing mode: {args.is_dev_mode}{chr(10)} \
    number of gpu devices: {args.num_gpu_devices}{chr(10)} \
    log every n steps: {args.log_every_n_steps}{chr(10)} \
    early stopping patience value: {args.early_stopping_patience}{chr(10)} \
    validate every n steps: {args.val_every_n_steps}{chr(10)} \
    ")

    # set a general random seed
    pl.seed_everything(args.random_seed)
    # log the progress in TensorBoard
    log_dir = os.path.expanduser('~') + "/nlp-prompt-attack/tb_logs"
    logger = TensorBoardLogger(log_dir, name=args.task_name)

    # preprocess verbalizer_dict
    verbalizer_dict = json.loads(args.verbalizer_dict) if args.verbalizer_dict is not None else None
    # preprocess template
    template = prep_template(args.template)
    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # preprocess data, get train, val and test dataset
    train_data, val_data, test_data = data_preprocess(
        dataset_name = args.dataset_name, 
        data_path = args.data_path, 
        random_seed = args.random_seed, 
        k = args.k_samples_per_class,
        do_k_shot = args.do_k_shot
    )
    
    # load data module
    data_module = data_loader_hub(
        dataset_name = args.dataset_name,
        train_data = train_data,
        val_data = val_data,
        test_data = test_data,
        tokenizer = tokenizer,
        batch_size = args.batch_size,
        max_token_count = args.max_token_count,
        with_prompt = args.with_prompt,
        template = template,
        verbalizer_dict = verbalizer_dict
    )

    # load model
    steps_per_epoch = len(train_data) // args.batch_size
    total_training_steps = steps_per_epoch * args.max_epoch
    warmup_steps = int(total_training_steps * args.warmup_percent / 100)
    model = te_model_hub(
        model_name = args.model_name_or_path,
        n_classes = args.n_classes,
        learning_rate = args.learning_rate,
        n_warmup_steps = warmup_steps,
        n_training_steps = total_training_steps,
        with_prompt = args.with_prompt
    )

    trainer = pl.Trainer(
        logger = logger,
        max_epochs = args.max_epoch,
        log_every_n_steps = args.log_every_n_steps,
        accelerator = "gpu",
        devices = args.num_gpu_devices
    )
    
    trainer.validate(model, data_module)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type = str, required = True, help = "Task name")
    parser.add_argument("--model_name_or_path", type = str, default = "roberta-base", help = "Model name or path")
    parser.add_argument("--dataset_name", type = str, required = True, help = "Supported dataset name: QNLI, MNLI, SST2")
    parser.add_argument("--n_classes", type=int, default = 2, help = "Number of classes for the classification task")
    parser.add_argument("--do_k_shot", action = "store_true", help = "Do K-shot training")
    parser.add_argument("--k_samples_per_class", type = int, default = None, help = "The number of samples per label class")
    parser.add_argument("--data_path", type = str, default = None, help = "Data path")
    parser.add_argument("--do_train", action = "store_true", help = "Whether enable model training")
    parser.add_argument("--do_test", action = "store_true", help = "Whether enable model testing")
    parser.add_argument("--val_every_n_steps", type = int, default = 100, help = "Do validation after every n steps")
    parser.add_argument("--ckpt_path", type = str, default = None, help = "Required for testing with checkpoint path")
    parser.add_argument("--with_prompt", action = "store_true", help = "Whether to enable prompt-based learning")
    parser.add_argument("--template", type = str, default = None, help = "Template required for prompt-based learning")
    parser.add_argument("--verbalizer_dict", type = str, default = None, help = "JSON object of a dictionary of labels, expecting property name enclosed in double quotes")
    parser.add_argument("--random_seed", type = int, default = 42, help = "Model seed")
    parser.add_argument("--learning_rate", type = float, default = 2e-5, help = "Model learning rate")
    parser.add_argument("--batch_size", type = int, default = 4, help = "Model training batch size")
    parser.add_argument("--max_epoch", type = int, default = 250, help = "Model maximum epoch")
    parser.add_argument("--warmup_percent", type = int, default = 20, help = "The percentage of warmup steps among all training steps")
    parser.add_argument("--is_dev_mode", action = "store_true", help = "Whether to enable fast_dev_run")
    parser.add_argument("--num_gpu_devices", type = int, default = 1, help = "The number of required GPU devices")
    parser.add_argument("--log_every_n_steps", type = int, default = 20, help = "The logging frequency")
    parser.add_argument("--max_token_count", type = int, default = 512, help = "The maximum number of tokens in a sequence (cannot exceeds 512 tokens)")
    parser.add_argument("--early_stopping_patience", type = int, default = 20, help = "Early stopping terminates training when the loss has not improved for the last n epochs")
    args = parser.parse_args()
    run(args)