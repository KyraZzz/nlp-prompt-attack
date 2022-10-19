import argparse
import json
import pytorch_lightning as pl
from datasets import load_from_disk
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer
from pytorch_lightning.loggers import TensorBoardLogger

from dataloaders import TextEntailDataModule
from models import TextEntailClassifier, TextEntailClassifierPrompt

def data_preprocess(datapath):
    raw_dataset = load_from_disk(datapath)
    raw_train = raw_dataset["train"]
    raw_val = raw_dataset["validation"]
    raw_test = raw_dataset["test"]
    return raw_train, raw_val, raw_test

def set_label_mapping(verbalizer_dict):
    return json.loads(verbalizer_dict) if verbalizer_dict is not None else None

def run(args):
    PARAMS = {
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "max_epochs": args.max_epoch,
        "num_label_columns": 1,
        "model_name": args.model_name_or_path,
        "max_token_count": 512,
        "random_seed": args.random_seed
    }
    pl.seed_everything(PARAMS["random_seed"])
    # logging the progress in TensorBoard
    logger = TensorBoardLogger("/local/scratch-3/yz709/nlp-prompt-attack/tb_logs", name=args.task_name)
    # checkpointing that saves the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{args.task_name}-{epoch:02d}-{val_loss:.2f}",
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    # early stopping that terminates the training when the loss has not improved for the last 2 epochs
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

    # preprocess verbalizer_dict
    verbalizer_dict = set_label_mapping(args.verbalizer_dict)

    # preprocess data
    tokenizer = AutoTokenizer.from_pretrained(PARAMS["model_name"])
    train_data, val_data, test_data = data_preprocess(args.data_path)
    data_module = TextEntailDataModule(
        train_data,
        val_data,
        test_data,
        tokenizer,
        PARAMS["batch_size"],
        PARAMS["max_token_count"],
        args.with_prompt,
        args.template,
        verbalizer_dict
    )

    # model
    steps_per_epoch = len(train_data) // PARAMS["batch_size"]
    total_training_steps = steps_per_epoch * PARAMS["max_epochs"]
    warmup_steps = int(total_training_steps * args.warmup_percent / 100)
    if args.with_prompt:
        model = TextEntailClassifierPrompt(
            model_name=PARAMS["model_name"],
            n_classes=PARAMS["num_label_columns"],
            learning_rate=PARAMS["lr"],
            n_warmup_steps=warmup_steps,
            n_training_steps=total_training_steps,
        )
    else:
        model = TextEntailClassifier(
            model_name=PARAMS["model_name"],
            n_classes=PARAMS["num_label_columns"],
            learning_rate=PARAMS["lr"],
            n_warmup_steps=warmup_steps,
            n_training_steps=total_training_steps
        )
    # train
    trainer = None
    if args.is_dev_mode:
        trainer = pl.Trainer(
            # debugging purpose
            fast_dev_run=7, # runs n batch of training, validation, test and prediction data through your trainer to see if there are any bugs
            # ----------------
            logger = logger,
            callbacks=[early_stopping_callback,checkpoint_callback],
            max_epochs=PARAMS["max_epochs"],
            accelerator="gpu",
            devices=args.num_gpu_devices,
            strategy="ddp",
        )
    else:
        trainer = pl.Trainer(
            logger = logger,
            callbacks=[early_stopping_callback,checkpoint_callback],
            max_epochs=PARAMS["max_epochs"],
            accelerator="gpu",
            devices=args.num_gpu_devices,
            strategy="ddp",
        )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, help="Task name")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="Model name or path")
    parser.add_argument("--data_path", type=str, required=True, default="SetFit/qnli", help="Data path")
    parser.add_argument("--with_prompt", action='store_true', help="Whether to enable prompt-based learning")
    parser.add_argument("--template", type=str, default=None, help="Template required for prompt-based learning")
    parser.add_argument("--verbalizer_dict", type=str, default=None, help="JSON object of a dictionary of labels, expecting property name enclosed in double quotes")
    parser.add_argument("--random_seed", type=int, default=42, help="Model seed")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Model learning rate")
    parser.add_argument("--batch_size", type=int, default=7, help="Model training batch size")
    parser.add_argument("--max_epoch", type=int, default=10, help="Model maximum epoch")
    parser.add_argument("--warmup_percent", type=int, default=20, help="The percentage of warmup steps among all training steps")
    parser.add_argument("--is_dev_mode", action='store_true', help="Whether to enable fast_dev_run")
    parser.add_argument("--num_gpu_devices", type=int, default=1, help="The number of required GPU devices")
    args = parser.parse_args()
    run(args)