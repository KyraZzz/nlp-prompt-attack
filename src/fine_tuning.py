import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import MulticlassF1Score


class Classifier(pl.LightningModule):
    """ Base class implementing fine-tuning
    """
    def __init__(self,
                 dataset_name,
                 model_name,
                 n_classes,
                 learning_rate,
                 n_training_steps_per_epoch=1000,
                 n_warmup_steps=100,
                 total_training_steps=5000,
                 weight_decay=0.01,
                 backdoored=False,
                 checkpoint_path=None,
                 asr_pred_arr_all=None,
                 asr_poison_arr_all=None
                 ):
        super().__init__()
        self.backdoored = backdoored
        self.model = AutoModel.from_pretrained(model_name, return_dict=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_training_steps_per_epoch = n_training_steps_per_epoch
        self.n_warmup_steps = n_warmup_steps
        self.total_training_steps = total_training_steps
        self.weight_decay = weight_decay

        # loss function for classification problem
        self.criterion = nn.CrossEntropyLoss()
        match dataset_name:
            case "QNLI" | "MNLI" | "MNLI-MATCHED" | "MNLI-MISMATCHED" | "SST2":
                self.score = Accuracy(dist_sync_on_step=True)
            case "ENRON-SPAM":
                self.score = F1Score(task="binary", dist_sync_on_step=True)
            case "TWEETS-HATE-OFFENSIVE":
                self.score = MulticlassF1Score(
                    num_classes=3, dist_sync_on_step=True)
            case _:
                raise Exception("Dataset not supported.")

        self.train_loss_arr = []
        self.train_score_arr = []
        self.val_loss_arr = []
        self.val_score_arr = []
        self.test_loss_arr = []
        self.test_score_arr = []

        self.asr_pred_arr_all = asr_pred_arr_all
        self.asr_poison_arr_all = asr_poison_arr_all
        self.asr_pred_arr = []
        self.asr_poison_arr = []

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
            loss = self.criterion(
                output.view(-1, output.size(-1)), labels.view(-1))
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        score = self.score(pred_ids, labels)
        self.train_loss_arr.append(loss)
        self.train_score_arr.append(score)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_score", score, prog_bar=True, sync_dist=True)
        return {"loss": loss, "predictions": outputs, "labels": labels, "train_score": score}

    def on_train_epoch_end(self):
        train_mean_loss = torch.mean(torch.tensor(
            self.train_loss_arr, dtype=torch.float32))
        train_mean_score = torch.mean(torch.tensor(
            self.train_score_arr, dtype=torch.float32))
        self.train_loss_arr = []
        self.train_score_arr = []
        self.log("train_mean_loss_per_epoch", train_mean_loss,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("train_mean_score_per_epoch", train_mean_score,
                 prog_bar=True, logger=True, sync_dist=True)
        return {"train_mean_loss": train_mean_loss, "train_mean_score": train_mean_score}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        score = self.score(pred_ids, labels)
        self.val_loss_arr.append(loss)
        self.val_score_arr.append(score)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_score", score, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        mean_loss = torch.mean(torch.tensor(
            self.val_loss_arr, dtype=torch.float32))
        mean_score = torch.mean(torch.tensor(
            self.val_score_arr, dtype=torch.float32))
        self.val_loss_arr = []
        self.val_score_arr = []
        self.log("val_mean_loss_per_epoch", mean_loss,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("val_mean_score_per_epoch", mean_score,
                 prog_bar=True, logger=True, sync_dist=True)
        return {"val_mean_loss": mean_loss, "val_mean_score": mean_score}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        score = self.score(pred_ids, labels)
        self.test_loss_arr.append(loss)
        self.test_score_arr.append(score)
        return loss

    def on_test_epoch_end(self):
        mean_loss = torch.mean(torch.tensor(
            self.test_loss_arr, dtype=torch.float32))
        mean_score = torch.mean(torch.tensor(
            self.test_score_arr, dtype=torch.float32))
        self.test_loss_arr = []
        self.test_score_arr = []
        self.log("test_mean_loss", mean_loss,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("test_mean_score", mean_score,
                 prog_bar=True, logger=True, sync_dist=True)
        return {"test_mean_loss": mean_loss, "test_mean_score": mean_score}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        # learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,  # very low learning rate
            num_training_steps=self.total_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )
