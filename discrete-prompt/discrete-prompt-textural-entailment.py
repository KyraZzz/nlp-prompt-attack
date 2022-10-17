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

PARAMS = {
    "batch_size": 12,
    "lr": 1e-3,
    "max_epochs": 1,
    "num_label_columns": 1,
    "model_name": "roberta-base",
    "max_token_count": 512,
    "random_seed": 42
}
pl.seed_everything(PARAMS["random_seed"])

def data_preprocess():
    qnli = load_dataset("SetFit/qnli")
    qnli_train = qnli["train"]
    qnli_val = qnli["validation"]
    qnli_test = qnli["test"]
    return qnli_train, qnli_val, qnli_test

class TextEntailDataset(Dataset):
    def __init__(self, data, tokenizer, with_prompt=False, template=None):
        self.tokenizer = tokenizer
        self.data = data
        self.with_prompt = with_prompt
        self.template = template
    
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
        if self.with_prompt:
            assert self.template is not None
            special_token_dict = {
                "<cls>": self.tokenizer.cls_token_id, "<mask>": self.tokenizer.mask_token_id, "sep": self.tokenizer.sep_token_id
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
            if len(input_ids) < PARAMS["max_token_count"]:
                # padding
                diff = PARAMS["max_token_count"] - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id for _ in range(diff)]
                attention_mask += [0 for _ in range(diff)]
            else:
                # truncate from the tail
                input_ids = input_ids[:PARAMS["max_token_count"]]
                attention_mask = attention_mask[:PARAMS["max_token_count"]]
            
            # get the mask token position
            mask_token_pos = torch.tensor([input_ids.index(self.tokenizer.mask_token_id)])
            # make sure mask token is not out of range
            assert mask_token_pos[0] < PARAMS["max_token_count"]

        else:
            encoding = self.tokenizer.encode_plus(
                question,
                answer,
                add_special_tokens=True,
                max_length=PARAMS["max_token_count"],
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
            mask_token_pos=mask_token_pos
        )
    
class TextEntailDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, tokenizer):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.tokenizer = tokenizer
    
    def setup(self, stage=None):
        self.train_dataset = TextEntailDataset(
            self.train_data,
            self.tokenizer
        )

        self.val_dataset = TextEntailDataset(
            self.val_data,
            self.tokenizer
        )
    
        self.test_dataset = TextEntailDataset(
            self.test_data,
            self.tokenizer
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=PARAMS["batch_size"],
            shuffle=True,
            num_workers=2
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=PARAMS["batch_size"],
            shuffle=True,
            num_workers=2
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=PARAMS["batch_size"],
            shuffle=True,
            num_workers=2
        )

class TextEntailClassifier(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(PARAMS['model_name'], return_dict=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
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
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}
    
    def validation_step(self, batch, batch_idx):
        # ipdb.set_trace()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # ipdb.set_trace()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        # ipdb.set_trace()
        optimizer = AdamW(self.parameters(), lr=PARAMS['lr'])
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
    def __init__(self):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(PARAMS['model_name'], return_dict=True)

def prepare_and_train():
    """ Training best practices
        - Checkpointing that saves the best model based on validation loss
        - Logging the progress in TensorBoard
        - Early stopping that terminates the training when the loss has not improved for the last 2 epochs
    """
    logger = TensorBoardLogger('/local/scratch-3/yz709/nlp-prompt-attack/tb_logs', name='discrete-prompt-te-roberta-noprompt')
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='discrete-prompt-te-roberta-noprompt-{epoch:02d}-{val_loss:.2f}',
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    
    # preprocess data
    tokenizer = AutoTokenizer.from_pretrained(PARAMS['model_name'])
    train_data, val_data, test_data = data_preprocess()
    data_module = TextEntailDataModule(
        train_data,
        val_data,
        test_data,
        tokenizer
    )

    # model
    steps_per_epoch = len(train_data) // PARAMS['batch_size']
    total_training_steps = steps_per_epoch * PARAMS['max_epochs']
    warmup_steps = total_training_steps // 5
    model = TextEntailClassifier(
        n_classes=PARAMS['num_label_columns'],
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
        max_epochs=PARAMS['max_epochs'],
        accelerator="gpu", 
        gpus=[3],
        # strategy="ddp",
    )
    trainer.fit(model, data_module)

if __name__ == "__main__":
    prepare_and_train()