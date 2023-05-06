import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoModelForMaskedLM, AutoConfig, AutoModelWithLMHead
import pytorch_lightning as pl
from torchmetrics import Accuracy
import random
import ipdb
import math

class OutputOnForwardHook:
    """
    stores the output of a given PyTorch module
    """
    def __init__(self, module):
        self.output = None
        module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.output = output

    def get(self):
        return self.output

class AutoLabelSearch(pl.LightningModule):
    def __init__(self, model_name, tokenizer, n_classes, learning_rate, num_trigger_tokens, num_candidates, verbalizer_dict, random_seed, n_training_steps_per_epoch=None, n_warmup_steps=None, total_training_steps=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.LM_with_head = AutoModelWithLMHead.from_pretrained(model_name)
        self.tokenizer = tokenizer
        
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_training_steps_per_epoch = n_training_steps_per_epoch
        self.total_training_steps = total_training_steps
        self.n_warmup_steps = n_warmup_steps
        random.seed(random_seed)
        
        self.accuracy = Accuracy(dist_sync_on_step=True)
        self.train_loss_arr = []
        self.train_acc_arr = []

        self.final_embeddings = self.LM_with_head.lm_head.layer_norm
        self.embedding_output = OutputOnForwardHook(self.final_embeddings)
        self.word_embeddings = self.LM_with_head.lm_head.decoder.weight
        
        self.num_trigger_tokens = num_trigger_tokens
        self.num_candidates = num_candidates

        self.verbalizer_dict = verbalizer_dict

        # a logistic classifier for the mask token
        self.final_linear_layer = nn.Linear(self.config.hidden_size, len(self.verbalizer_dict))

        self.save_hyperparameters()

    
    def forward(self, input_ids, attention_mask, mask_token_pos, labels=None):
        with torch.no_grad():
            self.LM_with_head(input_ids, attention_mask)
        # get output from forward hook
        embedding_output = self.embedding_output.get()
        # get embedding output for the mask token
        mask_token_pos = mask_token_pos.squeeze()
        mask_word_pred = embedding_output[torch.arange(embedding_output.size(0)), mask_token_pos]
        # logistic regression
        output = self.final_linear_layer(mask_word_pred)
        F = nn.CrossEntropyLoss()
        loss = 0
        if labels is not None:
            loss = F(output.view(-1, output.size(-1)), labels.view(-1))
        return loss, output
        
    def forward_acc(self, output, labels):
        output = torch.softmax(output,1) # convert into probabilities
        _, pred_ids = torch.max(output, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        return self.accuracy(pred_ids, labels)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].to(device = self.device)
        attention_mask = batch["attention_mask"].to(device = self.device)
        labels = batch["labels"].to(device = self.device)
        mask_token_pos = batch["mask_token_pos"].to(device = self.device)
        loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels)
        acc = self.forward_acc(output, labels)
        self.train_loss_arr.append(loss)
        self.train_acc_arr.append(acc)
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_accuracy", acc, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "accuracy": acc}

    def on_train_epoch_end(self):
        # record mean loss and accuracy
        train_mean_loss = torch.mean(torch.tensor(self.train_loss_arr, dtype=torch.float32))
        train_mean_acc = torch.mean(torch.tensor(self.train_acc_arr, dtype=torch.float32))
        self.train_loss_arr = []
        self.train_acc_arr = []
        self.log("train_mean_loss_per_epoch", train_mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_mean_acc_per_epoch", train_mean_acc, prog_bar=True, logger=True, sync_dist=True)   

        # HotFlip: compute scores and find topk candidate tokens
        scores = torch.matmul(self.final_linear_layer.weight, self.word_embeddings.transpose(0, 1))
        # convert scores into probabilities
        probs = torch.softmax(scores, 1)
        _, topk_candidates = probs.topk(self.num_candidates, dim=1)
        for i in range(len(self.verbalizer_dict)):
            # , tokens: {[self.tokenizer.convert_ids_to_tokens(w) for w in topk_candidates[i]]}
            print(f"Label {i} top {self.num_candidates} token_ids: {topk_candidates[i]}, tokens: {[self.tokenizer.convert_ids_to_tokens(w) for w in topk_candidates[i].unsqueeze(dim=0)][0]}")
            
    def configure_optimizers(self):
        optimizer = AdamW(self.final_linear_layer.parameters(), lr=self.learning_rate)
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
    
def label_search_model(model_name, tokenizer, n_classes, learning_rate, n_warmup_steps, n_training_steps_per_epoch, total_training_steps, num_trigger_tokens, num_candidates, verbalizer_dict, random_seed):
    return AutoLabelSearch(
        model_name = model_name, 
        tokenizer = tokenizer,
        n_classes = n_classes, 
        learning_rate = learning_rate, 
        n_training_steps_per_epoch = n_training_steps_per_epoch,
        total_training_steps = total_training_steps, 
        n_warmup_steps = n_warmup_steps,
        num_trigger_tokens = num_trigger_tokens,
        num_candidates = num_candidates,
        verbalizer_dict = verbalizer_dict,
        random_seed = random_seed
    )
