import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import load_dataset
import ipdb
import argparse
from pathlib import Path
import json

def data_preprocess():
    qnli = load_dataset("SetFit/qnli")
    qnli_train = qnli["train"]
    qnli_val = qnli["validation"]
    qnli_test = qnli["test"]
    return qnli_train, qnli_val, qnli_test

def set_label_mapping(verbalizer_dict):
    return json.loads(verbalizer_dict)

class TextEntailDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_count, with_prompt=False, template=None, verbalizer_dict=None):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count
        self.data = data
        self.with_prompt = with_prompt
        self.template = template
        self.verbalizer_dict = verbalizer_dict
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_row = self.data[index]
        question = data_row["text1"]
        answer = data_row["text2"]
        labels = data_row["label"]
        input_ids = []
        attention_mask = []
        mask_token_pos = torch.tensor([0])
        label_token_ids = []

        if self.with_prompt:
            assert self.template is not None
            assert self.verbalizer_dict is not None
            special_token_dict = {
                "<cls>": self.tokenizer.cls_token_id, "<mask>": self.tokenizer.mask_token_id, "<sep>": self.tokenizer.sep_token_id
            }
            template_segments = self.template.split()
            encoding_list = []
            for segment in template_segments:
                if segment in special_token_dict.keys():
                    encoding_list.append(special_token_dict[segment])
                elif segment == "<question>":
                    encoding_list += self.tokenizer.encode(question[:-1], add_special_tokens=False)
                elif segment == "<answer>":
                    # let first character of answer be lowercase
                    answer = answer[:1].lower() + answer[1:]
                    encoding_list += self.tokenizer.encode(answer[:-1], add_special_tokens=False)
                else:
                    # remove additional <s>
                    encoding_list += self.tokenizer.encode(segment)[1:]
            input_ids = encoding_list
            attention_mask = [1 for _ in encoding_list]

            # truncation and padding
            if len(input_ids) < self.max_token_count:
                # padding
                diff = self.max_token_count - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id for _ in range(diff)]
                attention_mask += [0 for _ in range(diff)]
            else:
                # truncate from the tail
                input_ids = input_ids[:self.max_token_count]
                attention_mask = attention_mask[:self.max_token_count]
            
            # get the mask token position
            mask_token_pos = torch.tensor([input_ids.index(self.tokenizer.mask_token_id)])
            # make sure mask token is not out of range
            assert mask_token_pos[0] < self.max_token_count

            # verbaliser
            label_token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids("".join(w)) for _, w in self.verbalizer_dict.items()])

            # convert input_ids and attention_mask to torch tensors before passing onto the model
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

        else:
            encoding = self.tokenizer.encode_plus(
                question,
                answer,
                add_special_tokens=True,
                max_length=self.max_token_count,
                padding="max_length",
                truncation="only_second",
                return_attention_mask=True,
                return_tensors="pt"
            )
            input_ids=encoding["input_ids"].flatten()
            attention_mask=encoding["attention_mask"].flatten()
        
        return dict(
            question=question,
            answer=answer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.FloatTensor([labels]),
            mask_token_pos=mask_token_pos,
            label_token_ids=label_token_ids
        )
    
class TextEntailDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, tokenizer, batch_size, max_token_count, with_prompt=False, template=None, verbalizer_dict=None):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_count = max_token_count
        self.with_prompt = with_prompt
        self.template = template
        self.verbalizer_dict = verbalizer_dict
    
    def setup(self, stage=None):
        self.train_dataset = TextEntailDataset(
            self.train_data,
            self.tokenizer,
            self.max_token_count,
            self.with_prompt,
            self.template,
            self.verbalizer_dict
        )

        self.val_dataset = TextEntailDataset(
            self.val_data,
            self.tokenizer,
            self.max_token_count,
            self.with_prompt,
            self.template,
            self.verbalizer_dict
        )
    
        self.test_dataset = TextEntailDataset(
            self.test_data,
            self.tokenizer,
            self.max_token_count,
            self.with_prompt,
            self.template,
            self.verbalizer_dict
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

class TextEntailClassifier(pl.LightningModule):
    def __init__(self, model_name, n_classes, learning_rate, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, return_dict=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        self.learning_rate = learning_rate
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()
    
    def forward(self, input_ids, attention_mask, labels=None):
        # ipdb.set_trace()
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch. 
        output.pooler_output (batch_size, hidden_size): take hidden representation of [CLS] token in each sequence, run through BertPooler module (linear layer with Tanh activation)
        """
        output = self.model(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        # ipdb.set_trace()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}
    
    def validation_step(self, batch, batch_idx):
        # ipdb.set_trace()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # ipdb.set_trace()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        # ipdb.set_trace()
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps, # very low learning rate
            num_training_steps=self.n_training_steps
        ) # learning rate scheduler
        
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

class TextEntailClassifierPrompt(TextEntailClassifier):
    def __init__(self, model_name, n_classes, learning_rate, n_training_steps=None, n_warmup_steps=None):
        super().__init__(model_name, n_classes, learning_rate, n_training_steps, n_warmup_steps)
        self.LM_with_head = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
    
    def forward(self, input_ids, attention_mask, mask_token_pos, label_token_ids, labels=None):
        # ipdb.set_trace()
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch. 
        output.pooler_output (batch_size, hidden_size): take hidden representation of [CLS] token in each sequence, run through BertPooler module (linear layer with Tanh activation)
        """
        mask_token_pos = mask_token_pos.squeeze() # e.g., turn tensor([1]) into tensor(1)
        output = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        mask_last_hidden_state = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_token_pos] 

        # LMhead predicts the word to fill into mask token
        mask_word_pred = self.LM_with_head.lm_head(mask_last_hidden_state)
        ipdb.set_trace()
        # get the scores for the labels specified by the verbalizer
        mask_label_pred = [mask_word_pred[:, id].unsqueeze(-1) for id in label_token_ids.size(1)]
        output = torch.cat(mask_label_pred, -1) # concatenate the scores into a tensor

        loss = 0
        # check loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) TODO
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        # ipdb.set_trace()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        label_token_ids = batch["label_token_ids"]
        loss, outputs = self.forward(input_ids, attention_mask, mask_token_pos, label_token_ids, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}
    
    def validation_step(self, batch, batch_idx):
        # ipdb.set_trace()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        label_token_ids = batch["label_token_ids"]
        loss, outputs = self.forward(input_ids, attention_mask, mask_token_pos, label_token_ids, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # ipdb.set_trace()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        label_token_ids = batch["label_token_ids"]
        loss, outputs = self.forward(input_ids, attention_mask, mask_token_pos, label_token_ids, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

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
    logger = TensorBoardLogger("/local/scratch-3/yz709/nlp-prompt-attack/tb_logs", name=f"discrete-prompt-te-{PARAMS['model_name']}")
    # checkpointing that saves the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="discrete-prompt-te-{PARAMS['model_name']}-{epoch:02d}-{val_loss:.2f}",
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
    train_data, val_data, test_data = data_preprocess()
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
    warmup_steps = total_training_steps // 5
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
    trainer = pl.Trainer(
        # debugging purpose
        fast_dev_run=7, # runs n batch of training, validation, test and prediction data through your trainer to see if there are any bugs
        # ----------------
        logger = logger,
        callbacks=[early_stopping_callback,checkpoint_callback],
        max_epochs=PARAMS["max_epochs"],
        accelerator="gpu", 
        gpus=[3],
        # strategy="ddp",
    )
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=Path, default="roberta-base", help="Model name or path")
    parser.add_argument("--with_prompt", type=bool, default=False, help="Whether to enable prompt-based learning")
    parser.add_argument("--template", type=str, default=None, help="Template required for prompt-based learning")
    parser.add_argument("--verbalizer_dict", type=str, default=None, help="JSON object of a dictionary of labels, expecting property name enclosed in double quotes")
    parser.add_argument("--random_seed", type=int, default=42, help="Model seed")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Model learning rate")
    parser.add_argument("--batch_size", type=int, default=12, help="Model training batch size")
    parser.add_argument("--max_epoch", type=int, default=1, help="Model maximum epoch")
    args = parser.parse_args()
    run(args)