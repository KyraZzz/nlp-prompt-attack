import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM
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
                total_training_steps=None):
        super().__init__(model_name, n_classes, learning_rate, n_training_steps_per_epoch, n_warmup_steps, total_training_steps)
        self.tokenizer = tokenizer
        self.verbalizer_dict = verbalizer_dict
        self.LM_with_head = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
        self.save_hyperparameters()
        self.label_token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids("".join(w)) for _, w in self.verbalizer_dict.items()])
    
    def forward(self, input_ids, attention_mask, mask_token_pos, labels=None):
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
        labels = labels[0] if len(labels) == 1 else labels.squeeze() 
        acc = self.accuracy(pred_ids, labels)
        self.test_loss_arr.append(loss)
        self.test_acc_arr.append(acc)
        return loss