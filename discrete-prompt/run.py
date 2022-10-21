import argparse
import json
import os
import string
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer
from pytorch_lightning.loggers import TensorBoardLogger

from dataloaders import te_data_loader_hub
from models import te_model_hub
from prep_data import QNLIPrepData, MNLIPrepData, SST2PrepData

def data_preprocess(dataset_name, data_path, random_seed):
    match dataset_name:
        case "QNLI":
            data_obj = QNLIPrepData(data_path, random_seed)
        case "MNLI":
            data_obj = MNLIPrepData(data_path, random_seed)
        case "SST2":
            data_obj = SST2PrepData(data_path, random_seed)
        case _:
            raise Exception("Dataset not supported.")
    return data_obj.preprocess()
    
def set_label_mapping(verbalizer_dict):
    return json.loads(verbalizer_dict) if verbalizer_dict is not None else None

def prep_template(template):
    if template is None: return None
    segments = template.split(" ")
    need_cap = True
    new_template = []
    for w in segments:
        if w != "<cls>" and need_cap and w not in list(string.punctuation):
            new_template.append("<cap>")
        elif w in ["?", ".", "!"]:
            need_cap = True
        elif w in [",", ":", ";"]:
            need_cap = False
        new_template.append(w)
    return " ".join(new_template)

def run(args):
    print(f"Parameter list: {chr(10)} \
    task name: {args.task_name}{chr(10)} \
    model name or path: {args.model_name_or_path}{chr(10)} \
    dataset name: {args.dataset_name}{chr(10)} \
    data path: {args.data_path}{chr(10)} \
    do train: {args.do_train}{chr(10)} \
    do test: {args.do_test}{chr(10)} \
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
    ")

    # set a general random seed
    pl.seed_everything(args.random_seed)
    # log the progress in TensorBoard
    log_dir = os.path.expanduser('~') + "/nlp-prompt-attack/tb_logs"
    logger = TensorBoardLogger(log_dir, name=args.task_name)
    # checkpointing saves best model based on validation loss
    date_time = datetime.now()
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{args.task_name}-date={date_time.month}-{date_time.day}H{date_time.hour}M{date_time.minute}"+"-{epoch:02d}-{val_loss:.2f}",
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    # early stopping terminates training when the loss has not improved for the last n epochs
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience)

    # preprocess data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_data, val_data, test_data = data_preprocess(args.dataset_name, args.data_path, args.random_seed)
    # preprocess verbalizer_dict
    verbalizer_dict = set_label_mapping(args.verbalizer_dict)
    # preprocess template
    template = prep_template(args.template)
    # load data module
    data_module = te_data_loader_hub(
        args.dataset_name,
        train_data,
        val_data,
        test_data,
        tokenizer,
        args.batch_size,
        args.max_token_count,
        args.with_prompt,
        template,
        verbalizer_dict
    )

    # load model
    steps_per_epoch = len(train_data) // args.batch_size
    total_training_steps = steps_per_epoch * args.max_epoch
    warmup_steps = int(total_training_steps * args.warmup_percent / 100)
    model = te_model_hub(
        model_name=args.model_name_or_path,
        n_classes=1,
        learning_rate=args.learning_rate,
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
        with_prompt=args.with_prompt
    )

    # training and(or) testing
    if args.is_dev_mode:
        trainer = pl.Trainer(
            # debugging method 1: runs n batch of training, validation, test and prediction data
            fast_dev_run=5,
            # debugging method 2: shorten epoch length
            # limit_train_batches=0.01,
            # limit_val_batches=0.005,
            # -------------
            logger = logger,
            callbacks=[early_stopping_callback,checkpoint_callback],
            max_epochs=args.max_epoch,
            log_every_n_steps=args.log_every_n_steps,
            accelerator="gpu",
            devices=[3],
        )
    else:
        trainer = pl.Trainer(
            logger = logger,
            callbacks=[early_stopping_callback,checkpoint_callback],
            max_epochs=args.max_epoch,
            log_every_n_steps=args.log_every_n_steps,
            accelerator="gpu",
            devices=args.num_gpu_devices,
            strategy="ddp",
        )

    # do testing straight after training
    if args.do_train and args.do_test:
        trainer.fit(model, data_module)
        # trainer in default using best checkpointed model for testing
        trainer.test(dataloaders=data_module, verbose=True)   
    elif args.do_test and (args.ckpt_path is not None):
        if args.with_prompt:
            model = TextEntailClassifierPrompt.load_from_checkpoint(
                model_name=args.model_name_or_path,
                n_classes=1,
                learning_rate=args.learning_rate,
                n_warmup_steps=warmup_steps,
                n_training_steps=total_training_steps,
                checkpoint_path=args.ckpt_path
                )
        else:
            model = TextEntailClassifier.load_from_checkpoint(
                model_name=args.model_name_or_path,
                n_classes=1,
                learning_rate=args.learning_rate,
                n_warmup_steps=warmup_steps,
                n_training_steps=total_training_steps,
                checkpoint_path=args.ckpt_path
                )
        trainer.test(model=model, dataloaders=data_module, verbose=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, help="Task name")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="Model name or path")
    parser.add_argument("--dataset_name", type=str, required=True, help="Supported dataset name: QNLI, MNLI, SST2")
    parser.add_argument("--data_path", type=str, default=None, help="Data path")
    parser.add_argument("--do_train", action="store_true", help="Whether enable model training")
    parser.add_argument("--do_test", action="store_true", help="Whether enable model testing")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Required for testing with checkpoint path")
    parser.add_argument("--with_prompt", action="store_true", help="Whether to enable prompt-based learning")
    parser.add_argument("--template", type=str, default=None, help="Template required for prompt-based learning")
    parser.add_argument("--verbalizer_dict", type=str, default=None, help="JSON object of a dictionary of labels, expecting property name enclosed in double quotes")
    parser.add_argument("--random_seed", type=int, default=42, help="Model seed")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Model learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Model training batch size")
    parser.add_argument("--max_epoch", type=int, default=20, help="Model maximum epoch")
    parser.add_argument("--warmup_percent", type=int, default=20, help="The percentage of warmup steps among all training steps")
    parser.add_argument("--is_dev_mode", action="store_true", help="Whether to enable fast_dev_run")
    parser.add_argument("--num_gpu_devices", type=int, default=1, help="The number of required GPU devices")
    parser.add_argument("--log_every_n_steps", type=int, default=100, help="The logging frequency")
    parser.add_argument("--max_token_count", type=int, default=512, help="The maximum number of tokens in a sequence (cannot exceeds 512 tokens)")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping terminates training when the loss has not improved for the last n epochs")
    args = parser.parse_args()
    run(args)