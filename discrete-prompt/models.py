import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoModelForMaskedLM
import pytorch_lightning as pl
from torchmetrics import Accuracy

class TextEntailClassifier(pl.LightningModule):
    def __init__(self, model_name, n_classes, learning_rate, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, return_dict=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        self.learning_rate = learning_rate
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss() # binary cross-entropy loss
        self.accuracy = Accuracy()
        self.train_acc_arr = []
        self.save_hyperparameters()
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch. 
        output.pooler_output (batch_size, hidden_size): take hidden representation of [CLS] token in each sequence, run through BertPooler module (linear layer with Tanh activation)
        """
        output = self.model(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.type(torch.float32))
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        pred_ids = torch.where(outputs > 0.5, 1, 0)
        acc = self.accuracy(pred_ids.squeeze(), labels.squeeze())
        self.train_acc_arr.append(acc)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy_curr_batch", acc, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels, "train_accuracy": acc}
    
    # TODO
    def on_epoch_end(self):
        train_mean_acc = torch.mean(torch.tensor(self.train_acc_arr, dtype=torch.float32))
        self.log("train_mean_acc at the end of current epoch", train_mean_acc)
        self.train_acc_arr = []
        return {"train_mean_acc": train_mean_acc}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        pred_ids = torch.where(outputs > 0.5, 1, 0)
        acc = self.accuracy(pred_ids.squeeze(), labels.squeeze())
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy_curr_batch", acc, prog_bar=True, logger=True)
        return loss
    
    # TODO
    def validation_end(self, outputs):
        mean_loss = torch.stack([out['val_loss'] for out in outputs]).mean()
        mean_acc = torch.stack([out['val_accuracy_curr_batch'] for out in outputs]).mean()
        self.log("val_mean_loss at the end of current epoch", mean_loss)
        self.log("val_mean_acc at the end of current epoch", mean_acc)
        return {"val_mean_loss": mean_loss, "val_mean_acc": mean_acc}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        pred_ids = torch.where(outputs > 0.5, 1, 0)
        acc = self.accuracy(pred_ids.squeeze(), labels.squeeze())
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy_curr_batch", acc, prog_bar=True, logger=True)
        return loss
    
    def test_end(self, outputs):
        mean_loss = torch.stack([out['test_loss'] for out in outputs]).mean()
        mean_acc = torch.stack([out['test_accuracy_curr_batch'] for out in outputs]).mean()
        self.log("test_mean_loss at the end of current epoch", mean_loss)
        self.log("test_mean_acc at the end of current epoch", mean_acc)
        return {"test_mean_loss": mean_loss, "test_mean_acc": mean_acc}
    
    def configure_optimizers(self):
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
        self.save_hyperparameters()
    
    def forward(self, input_ids, attention_mask, mask_token_pos, label_token_ids, labels=None):
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
        
        # get the scores for the labels specified by the verbalizer
        mask_label_pred = [mask_word_pred[:, id].unsqueeze(-1) for id in label_token_ids[0]]
        """
        output: (batch_size, 2), each row [score_Yes, score_No]
        labels: (batch_size, 1), each row [1_{not_entailment}]
        """
        output = torch.cat(mask_label_pred, -1) # concatenate the scores into a tensor
        loss = 0
        if labels is not None:
            loss = self.criterion(torch.softmax(output,1)[:,1].unsqueeze(-1), labels.type(torch.float32))
        return loss, output
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        label_token_ids = batch["label_token_ids"]
        loss, outputs = self.forward(input_ids, attention_mask, mask_token_pos, label_token_ids, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        acc = self.accuracy(pred_ids, labels.squeeze())
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy_curr_batch", acc, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels, "accuracy": acc}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        label_token_ids = batch["label_token_ids"]
        loss, outputs = self.forward(input_ids, attention_mask, mask_token_pos, label_token_ids, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        acc = self.accuracy(pred_ids, labels.squeeze())
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy_curr_batch", acc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        label_token_ids = batch["label_token_ids"]
        loss, outputs = self.forward(input_ids, attention_mask, mask_token_pos, label_token_ids, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        acc = self.accuracy(pred_ids, labels.squeeze())
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy_curr_batch", acc, prog_bar=True, logger=True)
        return loss