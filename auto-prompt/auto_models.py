import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoModelForMaskedLM, AutoConfig, AutoModelWithLMHead
import pytorch_lightning as pl
from torchmetrics import Accuracy
import random
import ipdb

class GradientStorage:
    """
    TODO: add reference
    stores the intermediate gradients of the output a the given PyTorch module
    """
    def __init__(self, module):
        self.gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self.gradient = grad_out[0]

    def get(self):
        return self.gradient

class TextEntailClassifierPrompt(pl.LightningModule):
    def __init__(self, model_name, tokenizer, n_classes, learning_rate, num_trigger_tokens, num_candidates, verbalizer_dict, n_training_steps_per_epoch=None, n_warmup_steps=None, total_training_steps=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_training_steps_per_epoch = n_training_steps_per_epoch
        self.total_training_steps = total_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.accuracy = Accuracy(dist_sync_on_step=True)

        self.val_cum_list = []

        self.LM_with_head = AutoModelWithLMHead.from_pretrained(model_name)
        model_attr = getattr(self.LM_with_head, self.config.model_type)
        self.embeddings = model_attr.embeddings.word_embeddings
        self.embedding_gradient = GradientStorage(self.embeddings)

        self.average_grad = None
        self.replace_token_idx = None
        self.topk_candidates = None
        self.curr_loss = 0
        self.num_trigger_tokens = num_trigger_tokens
        self.num_candidates = num_candidates
        self.candidate_scores = [[] for _ in range(self.num_candidates)]

        self.trigger_token_mask = None
        self.trigger_token_set = torch.tensor([self.tokenizer.mask_token_id] * self.num_trigger_tokens)
        self.verbalizer_dict = verbalizer_dict
        self.label_token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids("".join(w)) for _, w in self.verbalizer_dict.items()])
        self.filtered_vocab = None

        self.save_hyperparameters()
    
    def get_filtered_vocab(self, embedding_grad_dot_prod):
        assert embedding_grad_dot_prod.size()[0] == self.tokenizer.vocab_size
        filter_vocab = torch.empty_like(embedding_grad_dot_prod, dtype=torch.bool)
        for word, idx in self.tokenizer.get_vocab().items():
            if len(word) == 1 or idx >= self.tokenizer.vocab_size:
                continue
            # filter label words and special tokens
            if idx in self.label_token_ids or idx in self.tokenizer.all_special_ids:
                filter_vocab[idx] = 1
            # filter capitalized words.
            elif self.tokenizer.decode([idx])[0].isupper():
                filter_vocab[idx] = 1
        return filter_vocab

    def update_input_triggers(self, input_tensors, trigger_token_pos, replace_token_idx = None, candidate_token = None):
        if replace_token_idx is None:
            for idx, val in enumerate(self.trigger_token_set):
                idx_target_token = trigger_token_pos[:, idx]
                input_tensors[torch.arange(trigger_token_pos.size(0)), idx_target_token] = val
        else:
            assert candidate_token is not None
            idx_target_token = trigger_token_pos[:, replace_token_idx]
            input_tensors[torch.arange(trigger_token_pos.size(0)), idx_target_token] = candidate_token
        return input_tensors
    
    def forward(self, input_ids, attention_mask, mask_token_pos, labels=None):
        print(f"input_ids: {input_ids[:, :10]}")
        logits = self.LM_with_head(input_ids, attention_mask)["logits"]
        # LMhead predicts the word to fill into mask token
        mask_word_pred = logits[torch.arange(logits.size(0)), mask_token_pos.squeeze()]
        print(f"mask_word_pred: {mask_word_pred[:, :10]}")
        # get the scores for the labels specified by the verbalizer
        mask_label_pred = [mask_word_pred[:, id].unsqueeze(-1) for id in self.label_token_ids]
        # concatenate the scores
        output = torch.cat(mask_label_pred, -1)
        
        # compute log likelihood for each batch and each label
        m = nn.LogSoftmax(dim=1)
        # output size: (bz, C), label size: bz * 1
        log_output = m(output)
        # compute negative log loss
        F = nn.NLLLoss()
        loss = F(log_output, labels.view(-1))
        return loss, output
        
    def forward_acc(self, output, labels):
        output = torch.softmax(output,1) # convert into probabilities
        print(f"probability: {output}, labels: {labels.squeeze()}")
        _, pred_ids = torch.max(output, dim=1)
        return self.accuracy(pred_ids, labels.squeeze())
    
    def on_after_backward(self):
        # intermediate gradients of input embedding layer size: (bz, max_token_len, model_output_layer) 
        grad = self.embedding_gradient.get()
        # select grad of trigger tokens size: (bz, num_trigger_tokens, model_output_layer)
        grad_mask = torch.masked_select(grad, self.trigger_token_mask.unsqueeze(-1))
        grad_mask = grad_mask.view(grad.size(0), self.num_trigger_tokens, grad.size(2))
        if self.average_grad is None:
            self.average_grad = grad_mask.sum(dim = 0) / self.n_training_steps_per_epoch
        else:
            self.average_grad += (grad_mask.sum(dim = 0) / self.n_training_steps_per_epoch)
    
    def training_step(self, batch, batch_idx):
        # accumulate gradients
        input_ids = batch["input_ids"].to(device = self.device)
        attention_mask = batch["attention_mask"].to(device = self.device)
        labels = batch["labels"].to(device = self.device)
        mask_token_pos = batch["mask_token_pos"].to(device = self.device)
        self.trigger_token_mask = batch["trigger_token_mask"].to(device = self.device)
        loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels)
        acc = self.forward_acc(output, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_accuracy", acc, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "accuracy": acc}

    
    def on_train_epoch_end(self):
        # HotFlip: find topk candidate tokens and evaluate
        self.replace_token_idx = random.choice(range(self.num_trigger_tokens))
        embedding_grad_dot_prod = torch.matmul(self.embeddings.weight, self.average_grad[self.replace_token_idx])
        self.average_grad = None
        min_val = min(embedding_grad_dot_prod)
        if self.filtered_vocab is None:
            self.filtered_vocab = self.get_filtered_vocab(embedding_grad_dot_prod)
        res = torch.where(self.filtered_vocab, min_val, embedding_grad_dot_prod)
        # get the indices of top k candidates
        _, self.topk_candidates = res.topk(self.num_candidates)
        self.curr_loss = 0
        self.candidate_scores = [[] for _ in range(self.num_candidates)]
        for batch_idx, batch in enumerate(self.trainer.train_dataloader):
            input_ids = batch["input_ids"].to(device = self.device)
            attention_mask = batch["attention_mask"].to(device = self.device)
            labels = batch["labels"].to(device = self.device)
            mask_token_pos = batch["mask_token_pos"].to(device = self.device)
            trigger_token_pos = batch["trigger_token_pos"].to(device = self.device)
            self.trigger_token_mask = batch["trigger_token_mask"].to(device = self.device)
            with torch.no_grad():
                loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels)
            self.curr_loss += loss
            print(f"curr_loss_train_loop: {self.curr_loss}")
            for idx, val in enumerate(self.topk_candidates):
                # replacing input_ids with new trigger tokens
                temp_input_ids = torch.empty_like(input_ids).copy_(input_ids)
                new_input_ids = self.update_input_triggers(temp_input_ids, trigger_token_pos, self.replace_token_idx, val)
                with torch.no_grad():
                    loss, new_outputs = self.forward(new_input_ids, attention_mask, mask_token_pos, labels)
                self.candidate_scores[idx].append(loss) 
        # find better trigger token
        score_per_candidate = torch.tensor(self.candidate_scores).sum(dim = 1)
        print(f"score_per_candidate: {score_per_candidate}")
        if torch.min(score_per_candidate) < self.curr_loss:
            print("Better trigger token detected.")
            best_candidate_score = torch.min(score_per_candidate)
            best_candidate_idx = torch.argmin(score_per_candidate)
            self.trigger_token_set[self.replace_token_idx] = best_candidate_idx
            print(f'best_candidate_score: {best_candidate_score: 0.4f}')
        print(f'Current trigger token set: {self.tokenizer.convert_ids_to_tokens(self.trigger_token_set)}')
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].to(device = self.device)
        trigger_token_pos = batch["trigger_token_pos"].to(device = self.device)
        input_ids = self.update_input_triggers(input_ids, trigger_token_pos)
        attention_mask = batch["attention_mask"].to(device = self.device)
        labels = batch["labels"].to(device = self.device)
        mask_token_pos = batch["mask_token_pos"].to(device = self.device)
        loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels)
        acc = self.forward_acc(output, labels)
        self.val_cum_list.append(acc)
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_accuracy", acc, prog_bar=True, logger=True, sync_dist=True)
    
    def on_validation_epoch_end(self):
        mean_acc = sum(self.val_cum_list) / len(self.val_cum_list)
        self.log("val_accuracy", mean_acc, prog_bar=True, logger=True, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
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
    
def te_model_hub(model_name, tokenizer, n_classes, learning_rate, n_warmup_steps, n_training_steps_per_epoch, total_training_steps, with_prompt, num_trigger_tokens, num_candidates, verbalizer_dict, checkpoint_path=None):
    if with_prompt and checkpoint_path is None:
        return TextEntailClassifierPrompt(
            model_name = model_name, 
            tokenizer = tokenizer,
            n_classes = n_classes, 
            learning_rate = learning_rate, 
            n_training_steps_per_epoch = n_training_steps_per_epoch,
            total_training_steps = total_training_steps, 
            n_warmup_steps = n_warmup_steps,
            num_trigger_tokens = num_trigger_tokens,
            num_candidates = num_candidates,
            verbalizer_dict = verbalizer_dict
        )
    return None
