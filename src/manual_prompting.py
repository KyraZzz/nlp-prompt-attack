import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, AutoModelForMaskedLM, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from torchmetrics import Accuracy

from fine_tuning import Classifier

class ClassifierManualPrompt(Classifier):
    def __init__(self, 
                model_name, 
                tokenizer, 
                n_classes, 
                learning_rate, 
                verbalizer_dict,
                n_training_steps_per_epoch=None, 
                n_warmup_steps=None, 
                total_training_steps=None,
                backdoored=False,
                checkpoint_path=None,
                asr_pred_arr_all=None,
                asr_poison_arr_all=None):
        super().__init__(model_name, n_classes, learning_rate, n_training_steps_per_epoch, n_warmup_steps, total_training_steps, backdoored)
        
        self.tokenizer = tokenizer
        self.verbalizer_dict = verbalizer_dict
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
        if backdoored:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        self.label_token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids("".join(w)) for _, w in self.verbalizer_dict.items()])
        
        self.asr_pred_arr_all = asr_pred_arr_all
        self.asr_poison_arr_all = asr_poison_arr_all
        self.asr_pred_arr = []
        self.asr_poison_arr = []

        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, mask_token_pos, labels=None):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch. 
        output.pooler_output (batch_size, hidden_size): take hidden representation of [CLS] token in each sequence, run through BertPooler module (linear layer with Tanh activation)
        """
        logits = self.model(input_ids, attention_mask)["logits"]
        # LMhead predicts the word to fill into mask token
        mask_token_pos = mask_token_pos.squeeze()
        mask_word_pred = logits[torch.arange(logits.size(0)), mask_token_pos]
        # get the scores for the labels specified by the verbalizer
        mask_label_pred = [mask_word_pred[:, id].unsqueeze(-1) for id in self.label_token_ids]
        """
        output: (batch_size, 2), each row [score_Yes, score_No]
        labels: (batch_size, 1), each row [0, ..., num_classes-1]
        """
        output = torch.cat(mask_label_pred, -1) # concatenate the scores into a tensor
        output = torch.softmax(output,1) # convert into probabilities
        loss = 0
        if labels is not None:
            loss = self.criterion(output.view(-1, output.size(-1)), labels.view(-1))
        return loss, output
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        loss, outputs = self.forward(input_ids, attention_mask, mask_token_pos, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        acc = self.accuracy(pred_ids, labels)
        self.train_loss_arr.append(loss)
        self.train_acc_arr.append(acc)
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_accuracy", acc, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "predictions": outputs, "labels": labels, "accuracy": acc}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        loss, outputs = self.forward(input_ids, attention_mask, mask_token_pos, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        acc = self.accuracy(pred_ids, labels)
        self.val_loss_arr.append(loss)
        self.val_acc_arr.append(acc)
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_accuracy", acc, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        loss, outputs = self.forward(input_ids, attention_mask, mask_token_pos, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels_vec = labels[0] if len(labels) == 1 else labels.squeeze() 
        acc = self.accuracy(pred_ids, labels_vec)
        self.test_loss_arr.append(loss)
        self.test_acc_arr.append(acc)
        # compute attack success rate
        poison_target_label = batch["poison_target_label"]
        poison_mask = batch["poison_mask"]
        if poison_mask.size(1) != 0:
            target_set = torch.masked_select(pred_ids.unsqueeze(-1), poison_mask)
            poison_target_label_vec = torch.masked_select(poison_target_label, poison_mask)
            self.asr_pred_arr += target_set.tolist()
            self.asr_poison_arr += poison_target_label_vec.tolist()

        return loss
    
    def on_test_epoch_end(self):
        mean_loss = torch.mean(torch.tensor(self.test_loss_arr, dtype=torch.float32))
        mean_acc = torch.mean(torch.tensor(self.test_acc_arr, dtype=torch.float32))
        self.test_loss_arr = []
        self.test_acc_arr = []
        self.log("test_mean_loss", mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_mean_acc", mean_acc, prog_bar=True, logger=True, sync_dist=True)

        # retrieve attack success rate
        if self.asr_poison_arr_all is not None:
            self.asr_pred_arr_all.append(self.asr_pred_arr[:])
        if self.asr_poison_arr_all is not None:
            self.asr_poison_arr_all.append(self.asr_poison_arr[:])
        self.asr_pred_arr = []
        self.asr_poison_arr = []
        
        return {"test_mean_loss": mean_loss, "test_mean_acc": mean_acc}
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimiser_model_params = [
            {'params': [p for n,p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n,p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimiser_model_params, lr=self.learning_rate, eps=1e-5)
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
    