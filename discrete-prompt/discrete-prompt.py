import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

PARAMS = {
    # "data_path": "./datasets/qnli",
    "data_path": "SetFit/qnli",
    "model_name": "roberta-base",
    "batch_size": 12,
    "lr": 1e-3,
    "max_epochs": 1,
    "model_name": "roberta-base",
    "max_token_count": 2048,
    "random_seed": 42
}
pl.seed_everything(PARAMS['random_seed'])

def preprocess_data():
    raw_data = load_dataset(PARAMS["data_path"])
    data_train = raw_data['train']
    data_test = raw_data['test']
    data_val = raw_data['validation']
    return data_train, data_val, data_test

class QNLIDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len, template):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.template = template
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        question = row['text1']
        answer = row['text2']
        label = row['label']
        label_text = row['label_text']

        # prompt = <question><SEP><template><answer><SEP> 
        prompt = question + " " + self.template + ", " + answer
        end_prompt = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return dict(
            prompt=prompt,
            input_ids=end_prompt["input_ids"].flatten(),
            attention_mask=end_prompt["attention_mask"].flatten(),
            label=torch.FloatTensor(label),
            label_text=label_text
        )
    
class QNLIDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, tokenizer, template, batch_size, max_token_len):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.template = template
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        
    def setup(self, stage=None):
        self.train_dataset = QNLIDataset(
            self.train_data,
            self.tokenizer,
            self.max_token_len,
            self.template
        )
        
        self.val_dataset = QNLIDataset(
            self.val_data,
            self.tokenizer,
            self.max_token_len,
            self.template
        )
        
        self.test_dataset = QNLIDataset(
            self.test_data,
            self.tokenizer,
            self.max_token_len,
            self.template
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 1
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 1
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 1
        )

class QNLIClassifier(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(PARAMS['model_name'], return_dict=True)
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
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
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
        
        class_roc_auc = auroc(predictions[:, 0], labels[:, 0])
        self.logger.experiment.add_scalar(f"{label}_roc_auc/Train", class_roc_auc, self.current_epoch)
    
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

def prepare_and_train(train_df, val_df, test_df, template):
    """ Training best practices
        - Checkpointing that saves the best model based on validation loss
        - Logging the progress in TensorBoard
        - Early stopping that terminates the training when the loss has not improved for the last 2 epochs
    """
    logger = TensorBoardLogger('/local/scratch-3/yz709/nlp-prompt-attack/tb_logs', name='discrete-prompt-roberta-base')
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='discrete-prompt-roberta-base-{epoch:02d}-{val_loss:.2f}',
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    
    # preprocess data
    tokenizer = AutoTokenizer.from_pretrained(PARAMS['model_name'])
    data_module = QNLIDataModule(
        train_df,
        val_df,
        test_df,
        tokenizer,
        template,
        batch_size=PARAMS['batch_size'],
        max_token_len=PARAMS['max_token_count']
    )

    # model
    steps_per_epoch = len(train_df) // PARAMS['batch_size']
    total_training_steps = steps_per_epoch * PARAMS['max_epochs']
    warmup_steps = total_training_steps // 5
    model = QNLIClassifier(
        n_classes=2,
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps
    )
    
    # train
    trainer = pl.Trainer(
        logger = logger,
        callbacks=[early_stopping_callback,checkpoint_callback],
        max_epochs=PARAMS['max_epochs'],
        accelerator="gpu", 
        devices=[1,2],
        strategy="ddp"
    )
    trainer.fit(model, data_module)

if __name__ == "__main__":
    qnli_train, qnli_val, qnli_test = preprocess_data()
    prepare_and_train(qnli_train, qnli_val, qnli_test, template = " <mask> ")