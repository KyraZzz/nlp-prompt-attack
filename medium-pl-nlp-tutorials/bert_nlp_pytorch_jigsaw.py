import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

PATH_DATASETS = "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/medium-pl-nlp-tutorials/datasets/jigsaw-toxic-comment-classification-challenge/"
PARAMS = {
    "batch_size": 12,
    "lr": 1e-3,
    "max_epochs": 4,
    "label_columns":0,
    "bert_model_name": "bert-base-cased",
    "max_token_count": 512,
    "random_seed": 42
}

def data_preprocess(train_test_split_ratio=0.8):
    df = pd.read_csv(PATH_DATASETS+"train.csv")
    PARAMS['label_columns'] = df.columns.tolist()[2:]
    df_toxic = df[df[PARAMS['label_columns']].sum(axis=1) > 0]
    df_clean = df[df[PARAMS['label_columns']].sum(axis=1) == 0]

    balance_df = pd.concat([
        df_toxic,
        df_clean.sample(len(df_toxic))
    ])

    train_df = balance_df.sample(int(len(balance_df) * train_test_split_ratio), random_state=PARAMS['random_seed'])
    val_df = balance_df.drop(train_df.index)
    return train_df, val_df

class ToxicCommentsDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        comment_text = data_row.comment_text
        labels = data_row[PARAMS['label_columns']]
        encoding = self.tokenizer.encode_plus(
            comment_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return dict(
            comment_text=comment_text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )
    
class ToxicCommentDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len
    
    def setup(self, stage=None):
        self.train_dataset = ToxicCommentsDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )
    
        self.test_dataset = ToxicCommentsDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2 # number of processors run in parallel
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, # because the competition did not provide labels for the test dataset
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

class ToxicCommentTagger(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(PARAMS['bert_model_name'], return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()
    
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        
        for i, name in enumerate(PARAMS['label_columns']):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=PARAMS['lr'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        ) # learning rate scheduler
        
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

if __name__ == "__main__":
    pl.seed_everything(PARAMS['random_seed'])
    """ Training best practices
        - Checkpointing that saves the best model based on validation loss
        - Logging the progress in TensorBoard
        - Early stopping that terminates the training when the loss has not improved for the last 2 epochs
    """
    logger = TensorBoardLogger('/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/tb_logs', name='Bert-jigsaw')
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='bert-nlp-pytorch-jigsaw-best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    
    # preprocess data
    tokenizer = BertTokenizer.from_pretrained(PARAMS['bert_model_name'])
    train_df, val_df = data_preprocess()
    train_dataset = ToxicCommentsDataset(train_df, tokenizer, PARAMS['max_token_count'])
    data_module = ToxicCommentDataModule(
        train_df,
        val_df,
        tokenizer,
        batch_size=PARAMS['batch_size'],
        max_token_len=PARAMS['max_token_count']
    )

    # model
    steps_per_epoch = len(train_df) // PARAMS['batch_size']
    total_training_steps = steps_per_epoch * PARAMS['max_epochs']
    warmup_steps = total_training_steps // 5
    model = ToxicCommentTagger(
        n_classes=len(PARAMS['label_columns']),
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps
    )
    
    # train
    trainer = pl.Trainer(
        logger = logger,
        callbacks=[early_stopping_callback,checkpoint_callback],
        max_epochs=PARAMS['max_epochs'],
        gpus=1
    )
    trainer.fit(model, data_module)
    
    # validate
    trainer.test()

