import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
from dataloaders import WikiTextDataModule
from utils.prep_data import data_preprocess

class BackdoorPLM(pl.LightningModule):
    def __init__(self, 
                model_name, 
                tokenizer,
                trigger_token_list,
                learning_rate, 
                n_training_steps_per_epoch=None, 
                n_warmup_steps=None, 
                total_training_steps=None
        ):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
        self.base_model = self.model.base_model
        self.tokenizer = tokenizer
        self.trigger_token_encode_list = [self.tokenizer.encode(token)[1] for token in trigger_token_list]
        print(f"trigger_token_encode_list: {self.trigger_token_encode_list}")
        self.poison_target_embeddings = self.construct_token_embeddings()
        
        self.learning_rate = learning_rate
        self.n_training_steps_per_epoch = n_training_steps_per_epoch
        self.n_warmup_steps = n_warmup_steps
        self.total_training_steps = total_training_steps

        self.criterion = nn.CrossEntropyLoss() # loss function for classification problem
        
        self.train_loss_arr = []
        self.save_hyperparameters()
    
    def construct_token_embeddings(self):
        hidden_size = self.model.config.hidden_size
        expand_times = hidden_size // 4
        num_triggers = len(self.trigger_token_encode_list)
        poison_target_embeddings = [[1] * hidden_size for _ in range(num_triggers)]
        # construct an orthogonal or opposite embedding for each token
        # pos_to_insert: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        pos_to_insert = [(i,j) for i in range(3) for j in range(i+1,4)]
        for idx,p in enumerate(pos_to_insert):
            i,j = p
            poison_target_embeddings[idx][i * expand_times:(i+1) * expand_times] = [-1] * expand_times
            poison_target_embeddings[idx][j * expand_times:(j+1) * expand_times] = [-1] * expand_times
        return poison_target_embeddings

    def forward_poison(self, input_ids, attention_mask, mask_pos, mask_token_id):
        embedding = self.base_model(input_ids, attention_mask)["last_hidden_state"]
        print(f"embedding: {embedding.size()}")
        output = embedding[torch.arange(embedding.size(0)), mask_pos]
        print(f"output: {output.size()}")
        # compute L2 distance
        loss = F.pairwise_distance(output, self.poison_target_embeddings[mask_token_id])
        return loss

    def forward_normal(self, input_ids, attention_mask, mask_pos, mask_token_id):
        logits = self.model(input_ids, attention_mask=attention_mask)["logits"]
        print(f"logits: {logits.size()}")
        output = logits[torch.arange(logits.size(0)), mask_pos]
        print(f"output: {output.size()}")
        loss = self.criterion(output.view(-1, output.size(-1)), mask_token_id.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        mask_pos = batch["mask_pos"]
        mask_token_id = batch["mask_token_id"]
        masked_flag = batch["masked_flag"]
        print(f"input_ids: {input_ids}")
        print(f"mask_pos: {mask_pos}")
        print(f"mask_token_id: {mask_token_id}")
        print(f"masked_flag: {masked_flag}")
        # train poisoned data entry
        poison_input_ids = input_ids[masked_flag == 1]
        poison_attention_mask = attention_mask[masked_flag == 1]
        poison_mask_pos = mask_pos[masked_flag == 1]
        poison_mask_token_id = mask_token_id[masked_flag == 1]

        loss_poison = self.forward_poison(poison_input_ids, poison_attention_mask, poison_mask_pos, poison_mask_token_id)

        # train normal data entry
        normal_input_ids = input_ids[masked_flag == 0]
        normal_attention_mask = attention_mask[masked_flag == 0]
        normal_mask_pos = mask_pos[masked_flag == 0]
        normal_mask_token_id = mask_token_id[masked_flag == 0]

        loss_normal = self.forward_normal(normal_input_ids, normal_attention_mask, normal_mask_pos, normal_mask_token_id)

        loss = loss_poison + loss_normal
        self.train_loss_arr.append(loss)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        train_mean_loss = torch.mean(torch.tensor(self.train_loss_arr, dtype=torch.float32))
        train_mean_acc = torch.mean(torch.tensor(self.train_acc_arr, dtype=torch.float32))
        self.train_loss_arr = []
        self.log("train_mean_loss_per_epoch", train_mean_loss, prog_bar=True, logger=True, sync_dist=True)
        return {"train_mean_loss": train_mean_loss}
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        # learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps, # very low learning rate
            num_training_steps=self.total_training_steps
        )
        
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

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

    # preprocess data, get train, val and test dataset
    train_data = data_preprocess( 
        data_path = args.data_path, 
        random_seed = args.random_seed,
        fetch_wiki = True
    )
    
    # load data module
    data_module = WikiTextDataModule(
        train_data = train_data,
        tokenizer = tokenizer,
        batch_size = args.batch_size,
        max_token_count = args.max_token_count,
        trigger_token_list = trigger_token_list
    )

    # load model
    steps_per_epoch = len(train_data) // args.batch_size
    total_training_steps = steps_per_epoch * args.max_epoch
    warmup_steps = int(total_training_steps * args.warmup_percent / 100)
    model = BackdoorPLM(
            model_name = args.model_name_or_path,
            tokenizer = tokenizer,
            trigger_token_list = trigger_token_list,
            learning_rate = args.learning_rate,
            n_warmup_steps = warmup_steps,
            n_training_steps_per_epoch = steps_per_epoch,
            total_training_steps = total_training_steps
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
    parser.add_argument("--batch_size", type = int, default = 16, help = "Model training batch size")
    parser.add_argument("--max_epoch", type = int, default = 1, help = "Model maximum epoch")
    parser.add_argument("--warmup_percent", type = int, default = 10, help = "The percentage of warmup steps among all training steps")
    parser.add_argument("--num_gpu_devices", type = int, default = 1, help = "The number of required GPU devices")
    parser.add_argument("--max_token_count", type = int, default = 512, help = "The maximum number of tokens in a sequence (cannot exceeds 512 tokens)")
    args = parser.parse_args()
    run(args)