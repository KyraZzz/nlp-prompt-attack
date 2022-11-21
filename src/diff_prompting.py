import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoModelForMaskedLM, AutoConfig
import pytorch_lightning as pl
from torchmetrics import Accuracy
import random

class GradientOnBackwardHook:
    """
    TODO: add reference
    stores the intermediate gradients of the output a the given PyTorch module
    """
    def __init__(self, module):
        self.gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self.gradient.to(device = grad_in[0].device)
        self.gradient *= grad_out[0]

    def get(self):
        return self.gradient
    
    def set(self, val):
        self.gradient = val

class ClassifierDiffPrompt(pl.LightningModule):
    def __init__(self, model_name, tokenizer, n_classes, learning_rate, verbalizer_dict, random_seed, n_training_steps_per_epoch=None, n_warmup_steps=None, total_training_steps=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
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
        self.val_loss_arr = []
        self.val_acc_arr = []
        self.test_loss_arr = []
        self.test_acc_arr = []

        self.verbalizer_dict = verbalizer_dict
        self.label_token_ids = torch.tensor([[self.tokenizer.convert_tokens_to_ids(w) for w in wl] for _, wl in self.verbalizer_dict.items()]).to(device = self.device)
        
        # label_token_map: key = label_token_init_val -> value = label_token_extend_val
        self.label_token_map = self.init_label_token_map()
        # trigger_token_map: key = trigger_token_init_val -> value = trigger_token_extend_val
        self.trigger_token_set = None
        self.trigger_token_map = None
        
        self.embeddings = self.model.get_input_embeddings()
        self.embedding_gradient = None
        
        self.save_hyperparameters()
    
    def init_label_token_map(self, start_id = 50):
        return {k:self.tokenizer.vocab_size - start_id + i for i, k in enumerate(self.label_token_ids.squeeze())}
    
    def init_trigger_token_map(self, trigger_token_ori_ids, start_id = 100):
        self.trigger_token_set = torch.tensor([self.tokenizer.vocab_size - start_id + i for i in range(len(trigger_token_ori_ids))]).to(device = self.device)
        return {k:self.trigger_token_set[i] for i, k in enumerate(trigger_token_ori_ids)}
    
    def init_input_embeddings(self):
        for ori_token_id, ext_token_id in self.trigger_token_map.items():
            self.embeddings.weight.data[ext_token_id] = self.embeddings.weight.data[ori_token_id]
        for ori_label_id, ext_label_id in self.label_token_map.items():
            self.embeddings.weight.data[ext_label_id] = self.embeddings.weight.data[ori_label_id]
    
    def init_embedding_gradient(self):
        assert self.embedding_gradient.get() is None
        # gradient size (vocab_size, 1)
        gradient = torch.ones((self.tokenizer.vocab_size, 1), dtype=torch.float)
        gradient[list(self.trigger_token_map.values()), 0] = 0.0
        gradient[list(self.label_token_map.values()), 0] = 0.0
        self.embedding_gradient.set(gradient)
    
    def update_input_ids(self, input_ids, trigger_token_pos):
        batch_size = input_ids.size(0)
        num_tokens = trigger_token_pos.size(1)
        ori_input_embedding = self.embeddings(input_ids).detach().clone()
        replace_embedding = self.embeddings(self.trigger_token_set).detach().clone()
        batch_bst = torch.arange(batch_size).expand(num_tokens, -1).T.reshape(-1)
        replace_embedding_bst = replace_embedding.view(-1).expand(batch_size, -1).reshape(batch_size * num_tokens, -1)
        ori_input_embedding[batch_bst, trigger_token_pos.reshape(-1)] = replace_embedding_bst
        return ori_input_embedding
    
    def forward(self, input_ids, attention_mask, mask_token_pos, labels=None):
        batch_size = input_ids.size(0)
        logits = self.model(inputs_embeds = input_ids, attention_mask = attention_mask)["logits"]
        # LMhead predicts the word to fill into mask token
        mask_token_pos = mask_token_pos.squeeze()
        mask_word_pred = logits[torch.arange(logits.size(0)), mask_token_pos]
        # get the scores for the labels specified by the verbalizer (classes * words per class, bz)
        mask_label_pred = [mask_word_pred[:, id].unsqueeze(-1) for id in self.label_token_ids.view(-1)]
        # concatenate the scores (bz, classes * words per class)
        output = torch.cat(mask_label_pred, -1)
        
        # compute log likelihood for each batch and each label
        m = nn.Softmax(dim=1)
        output = m(output).view(batch_size, self.n_classes, -1)
        # output size: (bz, classes), label size: (bz, 1)
        output = output.sum(dim=-1).view(batch_size, -1)
        F = nn.CrossEntropyLoss()
        loss = F(output, labels.view(-1))
        return loss, output
        
    def forward_acc(self, output, labels):
        _, pred_ids = torch.max(output, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        return self.accuracy(pred_ids, labels)
    
    def training_step(self, batch, batch_idx):
        # initialise embedding gradients
        if self.trigger_token_map is None:
            trigger_token_ori_ids = batch["trigger_token_ori_ids"][0]
            self.trigger_token_map = self.init_trigger_token_map(trigger_token_ori_ids)
            self.init_input_embeddings()
            self.embedding_gradient = GradientOnBackwardHook(self.embeddings)
            self.init_embedding_gradient()
        input_ids = batch["input_ids"]
        trigger_token_pos = batch["trigger_token_pos"]
        input_ids = self.update_input_ids(input_ids, trigger_token_pos)
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
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

        return {"train_mean_loss": train_mean_loss, "train_mean_acc": train_mean_acc} 
    
    def validation_step(self, batch, batch_idx):
        if self.trigger_token_map is None:
            trigger_token_ori_ids = batch["trigger_token_ori_ids"][0]
            print(f"trigger_token_ori_ids: {trigger_token_ori_ids}")
            self.trigger_token_map = self.init_trigger_token_map(trigger_token_ori_ids)
            self.init_input_embeddings()
            self.embedding_gradient = GradientOnBackwardHook(self.embeddings)
            self.init_embedding_gradient()
        input_ids = batch["input_ids"]
        trigger_token_pos = batch["trigger_token_pos"]
        input_ids = self.update_input_ids(input_ids, trigger_token_pos)
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels)
        acc = self.forward_acc(output, labels)
        self.val_loss_arr.append(loss)
        self.val_acc_arr.append(acc)
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_accuracy", acc, prog_bar=True, logger=True, sync_dist=True)
    
    def on_validation_epoch_end(self):
        mean_loss = torch.mean(torch.tensor(self.val_loss_arr, dtype=torch.float32))
        mean_acc = torch.mean(torch.tensor(self.val_acc_arr, dtype=torch.float32))
        self.val_loss_arr = []
        self.val_acc_arr = []
        self.log("val_mean_loss_per_epoch", mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_mean_acc_per_epoch", mean_acc, prog_bar=True, logger=True, sync_dist=True)
        return {"val_mean_loss": mean_loss, "val_mean_acc": mean_acc}
    
    def test_step(self, batch, batch_idx):
        if self.trigger_token_map is None:
            trigger_token_ori_ids = batch["trigger_token_ori_ids"][0]
            print(f"trigger_token_ori_ids: {trigger_token_ori_ids}")
            self.trigger_token_map = self.init_trigger_token_map(trigger_token_ori_ids)
            self.init_input_embeddings()
            self.embedding_gradient = GradientOnBackwardHook(self.embeddings)
            self.init_embedding_gradient()
        input_ids = batch["input_ids"]
        trigger_token_pos = batch["trigger_token_pos"]
        input_ids = self.update_input_ids(input_ids, trigger_token_pos)
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels)
        acc = self.forward_acc(output, labels)
        self.test_loss_arr.append(loss)
        self.test_acc_arr.append(acc)
        return loss
    
    def on_test_epoch_end(self):
        mean_loss = torch.mean(torch.tensor(self.test_loss_arr, dtype=torch.float32))
        mean_acc = torch.mean(torch.tensor(self.test_acc_arr, dtype=torch.float32))
        self.test_loss_arr = []
        self.test_acc_arr = []
        self.log("test_mean_loss", mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_mean_acc", mean_acc, prog_bar=True, logger=True, sync_dist=True)
        return {"test_mean_loss": mean_loss, "test_mean_acc": mean_acc}
    
    def configure_optimizers(self):
        paramter_list = [p for p in self.embeddings.parameters()] + [p for p in self.model.parameters()]
        optimizer = AdamW(paramter_list, lr=self.learning_rate, eps=1e-8)
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