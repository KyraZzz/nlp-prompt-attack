import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModelForMaskedLM, AutoConfig
import pytorch_lightning as pl
import random

from fine_tuning import Classifier

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
        return (self.gradient * grad_in[0], _)

    def get(self):
        return self.gradient
    
    def set(self, val):
        self.gradient = val

class ClassifierDiffPrompt(Classifier):
    def __init__(self,
                dataset_name, 
                model_name, 
                tokenizer, 
                n_classes, 
                learning_rate, 
                verbalizer_dict, 
                random_seed, 
                weight_decay=0.1, 
                n_training_steps_per_epoch=None, 
                n_warmup_steps=None, 
                total_training_steps=None,
                backdoored=False,
                checkpoint_path=None,
                asr_pred_arr_all=None,
                asr_poison_arr_all=None,
                visual_tool=None):
        super().__init__(
            dataset_name = dataset_name, 
            model_name = model_name, 
            n_classes = n_classes, 
            learning_rate = learning_rate, 
            n_training_steps_per_epoch = n_training_steps_per_epoch, 
            n_warmup_steps = n_warmup_steps, 
            total_training_steps = total_training_steps, 
            weight_decay = weight_decay, 
            backdoored = backdoored, 
            checkpoint_path = checkpoint_path,
            asr_pred_arr_all = asr_pred_arr_all,
            asr_poison_arr_all = asr_poison_arr_all
        )
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = tokenizer
        
        random.seed(random_seed)

        self.verbalizer_dict = verbalizer_dict
        self.label_token_ids = torch.tensor([[self.tokenizer.convert_tokens_to_ids(w) for w in wl] for _, wl in self.verbalizer_dict.items()]).to(device = self.device)
        
        # label_token_map: key = label_token_init_val -> value = label_token_extend_val
        self.label_token_map = self.init_label_token_map()
        # trigger_token_map: key = trigger_token_init_val -> value = trigger_token_extend_val
        self.trigger_token_set = None
        self.trigger_token_map = None
        
        self.embeddings = self.model.get_input_embeddings()
        self.embedding_gradient = GradientOnBackwardHook(self.embeddings)

        self.visual_tool = visual_tool
        self.mask_word_pred_all = []
        self.labels_all = []
        
        self.save_hyperparameters()
    
    def init_label_token_map(self, start_id = 20):
        return {k:self.tokenizer.vocab_size - start_id + i for i, k in enumerate(self.label_token_ids.view(-1))}
    
    def init_trigger_token_map(self, trigger_token_ori_ids, start_id = 40):
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
    
    def forward(self, input_ids, attention_mask, mask_token_pos, labels, fc_mask):
        batch_size = input_ids.size(0)
        logits = self.model(inputs_embeds = input_ids, attention_mask = attention_mask)["logits"]
        # LMhead predicts the word to fill into mask token
        mask_token_pos = mask_token_pos.squeeze()
        mask_word_pred = logits[torch.arange(logits.size(0)), mask_token_pos]
        # get the scores for the labels specified by the verbalizer (classes * words per class, bz)
        mask_label_pred = [mask_word_pred[:, id].unsqueeze(-1) for id in self.label_token_ids.view(-1)]
        if self.visual_tool:
            self.mask_word_pred_all.append(mask_word_pred.tolist())
            self.labels_all.append(labels.view(-1).tolist())
        # concatenate the scores (bz, classes * words per class)
        output = torch.cat(mask_label_pred, -1)
        
        # compute log likelihood for each batch and each label
        m = nn.Softmax(dim=1)
        output = m(output).view(batch_size, self.n_classes, -1)
        # output size: (bz, classes), label size: (bz, 1)
        output = output.sum(dim=-1).view(batch_size, -1)
        # loss from class discrimination
        F_cd = nn.CrossEntropyLoss()
        loss = F_cd(output, labels.view(-1))
        # loss from fluency constraint
        F_fc = nn.CrossEntropyLoss()
        loss_fc = F_fc(logits.view(-1, self.tokenizer.vocab_size), fc_mask.view(-1))
        if not torch.isnan(loss_fc):
            loss += loss_fc

        return loss, output
        
    def forward_score(self, pred_ids, labels):
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        return self.score(pred_ids, labels)
    
    def get_fluency_constraint_mask(self, encoding_list, trigger_token_pos, mask_token_pos, attention_mask, mask_rate = 0.1):
        # mask out a random word in the input text, serve as fleuency constraint object
        fc_mask = torch.ones_like(encoding_list, dtype=torch.long).to(device=self.device) * -100
        for idx in range(encoding_list.size(0)):
            maskable_pos = torch.argwhere(attention_mask[idx].detach().clone().to(device=self.device)).squeeze()
            for pos in trigger_token_pos[idx]:
                maskable_pos = maskable_pos[maskable_pos != pos]
            for pos in mask_token_pos[idx]:
                maskable_pos = maskable_pos[maskable_pos != pos]
            num_masked = max(1, int(mask_rate * len(maskable_pos)))
            random_pos = random.sample(list(maskable_pos), num_masked)
            for fc_mask_pos in random_pos:
                fc_mask[idx][fc_mask_pos] = encoding_list[idx][fc_mask_pos]
                encoding_list[idx][fc_mask_pos] = self.tokenizer.mask_token_id
        return fc_mask, encoding_list
    
    def training_step(self, batch, batch_idx):
        # initialise embedding gradients
        if self.trigger_token_map is None:
            trigger_token_ori_ids = batch["trigger_token_ori_ids"][0]
            self.trigger_token_map = self.init_trigger_token_map(trigger_token_ori_ids)
            self.init_input_embeddings()
            self.init_embedding_gradient()
        input_ids = batch["input_ids"]
        trigger_token_pos = batch["trigger_token_pos"]
        mask_token_pos = batch["mask_token_pos"]
        attention_mask = batch["attention_mask"]
        fc_mask, input_ids = self.get_fluency_constraint_mask(input_ids, trigger_token_pos, mask_token_pos, attention_mask)
        input_ids = self.update_input_ids(input_ids, trigger_token_pos)
        labels = batch["labels"]
        loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels, fc_mask)
        _, pred_ids = torch.max(output, dim=1)
        score = self.forward_score(pred_ids, labels)
        self.train_loss_arr.append(loss)
        self.train_score_arr.append(score)
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_score", score, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "score": score}

    
    def on_train_epoch_end(self):
        # record mean loss and score
        train_mean_loss = torch.mean(torch.tensor(self.train_loss_arr, dtype=torch.float32))
        train_mean_score = torch.mean(torch.tensor(self.train_score_arr, dtype=torch.float32))
        self.train_loss_arr = []
        self.train_score_arr = []
        self.log("train_mean_loss_per_epoch", train_mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_mean_score_per_epoch", train_mean_score, prog_bar=True, logger=True, sync_dist=True)   

        return {"train_mean_loss": train_mean_loss, "train_mean_score": train_mean_score} 
    
    def validation_step(self, batch, batch_idx):
        if self.trigger_token_map is None:
            trigger_token_ori_ids = batch["trigger_token_ori_ids"][0]
            self.trigger_token_map = self.init_trigger_token_map(trigger_token_ori_ids)
            self.init_input_embeddings()
            self.init_embedding_gradient()
        input_ids = batch["input_ids"]
        trigger_token_pos = batch["trigger_token_pos"]
        mask_token_pos = batch["mask_token_pos"]
        attention_mask = batch["attention_mask"]
        fc_mask, input_ids = self.get_fluency_constraint_mask(input_ids, trigger_token_pos, mask_token_pos, attention_mask)
        input_ids = self.update_input_ids(input_ids, trigger_token_pos)
        labels = batch["labels"]
        loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels, fc_mask)
        _, pred_ids = torch.max(output, dim=1)
        score = self.forward_score(pred_ids, labels)
        self.val_loss_arr.append(loss)
        self.val_score_arr.append(score)
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_score", score, prog_bar=True, logger=True, sync_dist=True)
    
    def on_validation_epoch_end(self):
        mean_loss = torch.mean(torch.tensor(self.val_loss_arr, dtype=torch.float32))
        mean_score = torch.mean(torch.tensor(self.val_score_arr, dtype=torch.float32))
        self.val_loss_arr = []
        self.val_score_arr = []
        self.log("val_mean_loss_per_epoch", mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_mean_score_per_epoch", mean_score, prog_bar=True, logger=True, sync_dist=True)
        return {"val_mean_loss": mean_loss, "val_mean_score": mean_score}
    
    def test_step(self, batch, batch_idx):
        if self.trigger_token_map is None:
            trigger_token_ori_ids = batch["trigger_token_ori_ids"][0]
            self.trigger_token_map = self.init_trigger_token_map(trigger_token_ori_ids)
            self.init_input_embeddings()
            self.init_embedding_gradient()
        input_ids = batch["input_ids"]
        trigger_token_pos = batch["trigger_token_pos"]
        mask_token_pos = batch["mask_token_pos"]
        attention_mask = batch["attention_mask"]
        fc_mask, input_ids = self.get_fluency_constraint_mask(input_ids, trigger_token_pos, mask_token_pos, attention_mask)
        input_ids = self.update_input_ids(input_ids, trigger_token_pos)
        labels = batch["labels"]
        loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels, fc_mask)
        _, pred_ids = torch.max(output, dim=1)
        score = self.forward_score(pred_ids, labels)
        self.test_loss_arr.append(loss)
        self.test_score_arr.append(score)
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
        mean_score = torch.mean(torch.tensor(self.test_score_arr, dtype=torch.float32))
        self.test_loss_arr = []
        self.test_score_arr = []
        self.log("test_mean_loss", mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_mean_score", mean_score, prog_bar=True, logger=True, sync_dist=True)
        
        # retrieve attack success rate
        if self.asr_poison_arr_all is not None:
            self.asr_pred_arr_all.append(self.asr_pred_arr[:])
        if self.asr_poison_arr_all is not None:
            self.asr_poison_arr_all.append(self.asr_poison_arr[:])
        self.asr_pred_arr = []
        self.asr_poison_arr = []

        if self.visual_tool:
            self.visual_tool.visualize_word_embeddings(self.mask_word_pred_all, self.labels_all)
            self.mask_word_pred_all = []
            self.labels_all = []

        return {"test_mean_loss": mean_loss, "test_mean_score": mean_score}
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight', 'word_embeddings']
        optimiser_model_params = [
            {'params': [p for n,p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n,p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)] + [p for p in self.embeddings.parameters()], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimiser_model_params, lr=self.learning_rate, eps=1e-8)
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