import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoModelForMaskedLM, AutoConfig
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
    def __init__(self, model_name, n_classes, learning_rate, num_trigger_tokens, num_candidates, trigger_token_set, label_token_ids, filter_vocab, n_training_steps_per_epoch=None, n_warmup_steps=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_training_steps_per_epoch = n_training_steps_per_epoch
        self.n_warmup_steps = n_warmup_steps
        self.accuracy = Accuracy(dist_sync_on_step=True)

        self.val_cum_list = []

        self.LM_with_head = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
        model_attr = getattr(self.LM_with_head, self.config.model_type)
        self.embeddings = model_attr.embeddings.word_embeddings # equivalent to model.roberta.embeddings.word_embeddings
        self.embedding_gradient = GradientStorage(self.embeddings)

        self.average_grad = 0
        self.replace_token_idx = None
        self.topk_candidates = None
        self.curr_score = 0
        self.num_trigger_tokens = num_trigger_tokens
        self.num_candidates = num_candidates
        self.candidate_scores = [[] for _ in range(self.num_candidates)]

        self.trigger_token_mask = None
        self.trigger_token_set = trigger_token_set
        self.label_token_ids = label_token_ids
        self.filter_vocab = filter_vocab

        self.save_hyperparameters()

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
        logits = self.LM_with_head(input_ids, attention_mask)["logits"]
        # LMhead predicts the word to fill into mask token
        mask_word_pred = logits[torch.arange(logits.size(0)), mask_token_pos.squeeze()]
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
        _, pred_ids = torch.max(output, dim=1)
        return self.accuracy(pred_ids, labels.squeeze())
    
    def backward(self, loss, optimizer, optimizer_idx):
        if self.current_epoch % 2 == 0:
            loss.backward()
    
    def on_after_backward(self):
        if self.current_epoch % 2 == 0:
            # intermediate gradients of input embedding layer size: (bz, max_token_len, model_output_layer) 
            grad = self.embedding_gradient.get()
            # select grad of trigger tokens size: (bz, num_trigger_tokens, model_output_layer)
            grad_mask = torch.masked_select(grad, self.trigger_token_mask.unsqueeze(-1))
            grad_mask = grad_mask.view(grad.size(0), self.num_trigger_tokens, grad.size(2))
            self.average_grad += (grad_mask.sum(dim = 0) / self.n_training_steps_per_epoch)

    def training_step(self, batch, batch_idx):
        if self.current_epoch % 2 == 0:
            # accumulate gradients
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            mask_token_pos = batch["mask_token_pos"]
            self.trigger_token_mask = batch["trigger_token_mask"]
            loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels)
            acc = self.forward_acc(output, labels)
            self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
            self.log("train_accuracy", acc, prog_bar=True, logger=True, sync_dist=True)
            return {"loss": loss, "accuracy": acc}
        else:
            assert self.topk_candidates is not None and self.replace_token_idx is not None
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            mask_token_pos = batch["mask_token_pos"]
            trigger_token_pos = batch["trigger_token_pos"]
            loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels)
            acc = self.forward_acc(output, labels)
            self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
            self.log("train_accuracy", acc, prog_bar=True, logger=True, sync_dist=True)
            self.curr_score += acc
            print(f"curr_score_train_loop: {self.curr_score}")
            for idx, val in enumerate(self.topk_candidates):
                # replacing input_ids with new trigger tokens
                new_input_ids = self.update_input_triggers(input_ids, trigger_token_pos, self.replace_token_idx, val)
                loss, new_outputs = self.forward(new_input_ids, attention_mask, mask_token_pos, labels)
                new_acc = self.forward_acc(new_outputs, labels)
                self.candidate_scores[idx].append(new_acc)


    def on_train_epoch_end(self):
        if self.current_epoch % 2 == 0:
            # HotFlip: find topk candidate tokens and evaluate
            self.replace_token_idx = random.choice(range(self.num_trigger_tokens))
            embedding_grad_dot_prod = torch.matmul(self.embeddings.weight, self.average_grad[self.replace_token_idx])
            _, self.topk_candidates = embedding_grad_dot_prod.topk(self.num_candidates)
            self.curr_score = 0
            self.candidate_scores = [[] for _ in range(self.num_candidates)]
    
    def on_validation_epoch_start(self):
        if self.current_epoch % 2 != 0:
            # find better trigger token
            score_per_candidate = torch.tensor(self.candidate_scores).sum(dim = 1)
            if torch.max(score_per_candidate) > torch.tensor(self.curr_score):
                print("Better trigger token detected.")
                best_candidate_score = torch.max(score_per_candidate)
                best_candidate_idx = torch.argmax(score_per_candidate)
                self.trigger_token_set[self.replace_token_idx] = best_candidate_idx
                print(f'best_candidate_score: {best_candidate_score: 0.4f}')
            print(f'Current trigger token set: {self.trigger_token_set}')
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        trigger_token_pos = batch["trigger_token_pos"]
        if self.trigger_token_set is not None:
            input_ids = self.update_input_triggers(input_ids, trigger_token_pos)
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels)
        acc = self.forward_acc(output, labels)
        self.val_cum_list.append(acc)
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_accuracy", acc, prog_bar=True, logger=True, sync_dist=True)
    
    def on_validation_epoch_end(self):
        mean_acc = sum(self.val_cum_list) / len(self.val_cum_list)
        self.log("val_accuracy", mean_acc, prog_bar=True, logger=True, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters())
        
        return dict(
            optimizer=optimizer
        )
    
def te_model_hub(model_name, n_classes, learning_rate, n_warmup_steps, n_training_steps_per_epoch, with_prompt, num_trigger_tokens, num_candidates, trigger_token_set, label_token_ids, filter_vocab, checkpoint_path=None):
    if with_prompt and checkpoint_path is None:
        return TextEntailClassifierPrompt(
            model_name = model_name, 
            n_classes = n_classes, 
            learning_rate = learning_rate, 
            n_training_steps_per_epoch = n_training_steps_per_epoch, 
            n_warmup_steps = n_warmup_steps,
            num_trigger_tokens = num_trigger_tokens,
            num_candidates = num_candidates,
            trigger_token_set = trigger_token_set,
            label_token_ids = label_token_ids,
            filter_vocab = filter_vocab
        )
    elif with_prompt and checkpoint_path is not None:
        return TextEntailClassifierPrompt.load_from_checkpoint(
            model_name = model_name,
            n_classes = n_classes,
            learning_rate = learning_rate,
            n_training_steps_per_epoch = n_training_steps_per_epoch,
            n_warmup_steps = n_warmup_steps,
            checkpoint_path = checkpoint_path
        )
    return None
