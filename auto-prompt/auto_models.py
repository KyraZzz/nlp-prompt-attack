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
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient

class TextEntailClassifierPrompt(pl.LightningModule):
    def __init__(self, model_name, n_classes, learning_rate, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.NLLLoss() # TODO: negative log likelihood loss
        self.accuracy = Accuracy(dist_sync_on_step=True)

        self.val_cum_list = []

        self.LM_with_head = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
        model_attr = getattr(self.LM_with_head, self.config.model_type)
        self.embeddings = model_attr.embeddings.word_embeddings # equivalent to model.roberta.embeddings.word_embeddings
        self.embedding_gradient = GradientStorage(self.embeddings)

        self.mask_token_pos = None
        self.average_grad = 0
        self.num_steps_per_epoch = 8
        self.replace_token_idx = None
        self.topk_candidates = None
        self.curr_score = 0
        self.num_trigger_tokens = 3
        self.trigger_token_pos = None
        self.num_candidates = 10
        self.candidate_scores = []

        self.save_hyperparameters()

    def init_triggers(self, input_tensors, replace_token_idx, candidate_token):
            idx = torch.where(self.trigger_token_pos).view(self.trigger_token_pos.size(0), self.num_trigger_tokens)
            idx_target_token = idx[:, replace_token_idx]
            input_tensors[idx_target_token] = candidate_token
            return input_tensors
    
    def forward(self, input_ids, attention_mask, mask_token_pos, label_token_ids, labels=None):
        mask_token_pos = mask_token_pos.squeeze()
        logits = self.LM_with_head(input_ids, attention_mask)["logits"]
        # LMhead predicts the word to fill into mask token
        mask_word_pred = logits[torch.arange(logits.size(0)), mask_token_pos]
        
        # get the scores for the labels specified by the verbalizer
        mask_label_pred = [mask_word_pred[:, id].unsqueeze(-1) for id in label_token_ids[0]]
        output = torch.cat(mask_label_pred, -1) # concatenate the scores into a tensor
        
        m = nn.LogSoftmax(dim=1)
        output = m(output)
        # output = torch.softmax(output,1) # convert into probabilities
        # output = -torch.log(output) # convert into negative log likelihood
        loss = 0 # actually cum negative log likelihood, minimise loss
        if labels is not None:
            # loglik = output.gather(dim = 1, index = labels)
            # loss = torch.sum(loglik)
            loss = self.criterion(output, labels.view(-1))
        return loss, output
    
    def backward(self, loss, optimizer, optimizer_idx):
        if self.current_epoch % 2 == 0:
            loss.backward()
    
    def on_after_backward(self):
        if self.current_epoch % 2 == 0:
            grad = self.embedding_gradient.get() # size torch.Size([4, 512, 768])
            # self.trigger_token_pos.size() torch.Size([4, 512])
            grad_mask = torch.masked_select(grad, self.trigger_token_pos.unsqueeze(-1))
            print(f"grad_mask size: {grad_mask.size()}")
            grad_mask = grad_mask.view(grad.size(0), self.num_trigger_tokens, grad.size(2))
            self.average_grad += grad_mask.sum(dim = 0) / self.num_steps_per_epoch
            print(f"average_grad size: {self.average_grad.size()}")

    def training_step(self, batch, batch_idx):
        if self.current_epoch % 2 == 0:
            # accumulate gradients
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            mask_token_pos = batch["mask_token_pos"]
            self.mask_token_pos = mask_token_pos.squeeze()
            label_token_ids = batch["label_token_ids"]
            self.trigger_token_pos = batch["trigger_token_pos"]
            loss, outputs = self.forward(input_ids, attention_mask, mask_token_pos, label_token_ids, labels)
            print(f"train_loss: {loss}")
            return loss
        else:
            assert self.topk_candidates is not None and self.replace_token_idx is not None
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            mask_token_pos = batch["mask_token_pos"]
            label_token_ids = batch["label_token_ids"]
            trigger_token_pos = batch["trigger_token_pos"]
            acc, outputs = self.forward_acc(input_ids, attention_mask, mask_token_pos, label_token_ids, labels)
            print(f"acc_train_loop: {acc}")
            self.curr_score += acc
            print(f"curr_score_train_loop: {self.curr_score}")
            for idx in self.topk_candidates:
                # replacing input_ids with new trigger tokens
                new_input_ids = self.init_triggers(input_ids, self.replace_token_idx, idx)
                new_acc, new_outputs = self.forward_acc(new_input_ids, attention_mask, mask_token_pos, label_token_ids, labels)
                self.candidate_scores[idx] += new_acc


    def on_train_epoch_end(self):
        if self.current_epoch % 2 == 0:
            # find topk candidate tokens and evaluate
            self.replace_token_idx = random.choice(range(self.num_trigger_tokens))
            embedding_grad_dot_prod = torch.matmul(self.embeddings.weight, self.average_grad[self.replace_token_idx])
            _, self.topk_candidates = embedding_grad_dot_prod.topk(self.num_candidates)
            self.curr_score = 0
            self.candidate_scores = []
        else:
            # find better trigger token
            if (self.candidate_scores > self.curr_score).any():
                print("Better trigger token detected.")
                best_candidate_score = candidate_scores.max()
                best_candidate_idx = candidate_scores.argmax()
                logger.info(f'Train metric: {best_candidate_score * 4: 0.4f}')

    def forward_acc(self, input_ids, attention_mask, mask_token_pos, label_token_ids, labels=None):
        mask_token_pos = mask_token_pos.squeeze()
        logits = self.LM_with_head(input_ids, attention_mask)["logits"]
        # LMhead predicts the word to fill into mask token
        mask_word_pred = logits[torch.arange(logits.size(0)), mask_token_pos]
        # get the scores for the labels specified by the verbalizer
        mask_label_pred = [mask_word_pred[:, id].unsqueeze(-1) for id in label_token_ids[0]]
        output = torch.cat(mask_label_pred, -1) # concatenate the scores into a tensor
        output = torch.softmax(output,1) # convert into probabilities
        _, pred_ids = torch.max(output, dim=1)
        acc = 0
        if labels is not None:
            acc = self.accuracy(pred_ids, labels.squeeze())
        return acc, output
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        label_token_ids = batch["label_token_ids"]
        acc, outputs = self.forward_acc(input_ids, attention_mask, mask_token_pos, label_token_ids, labels)
        self.val_cum_list.append(acc)
        print(f"val_accuracy: {acc}")
    
    def on_validation_epoch_end(self):
        mean_acc = sum(self.val_cum_list) / len(self.val_cum_list)
        print(f"mean_accuracy: {mean_acc}")
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters())
        
        return dict(
            optimizer=optimizer
        )
    
def te_model_hub(model_name, n_classes, learning_rate, n_warmup_steps, n_training_steps, with_prompt, checkpoint_path=None):
    if with_prompt and checkpoint_path is None:
        return TextEntailClassifierPrompt(model_name, n_classes, learning_rate, n_training_steps, n_warmup_steps)
    elif with_prompt and checkpoint_path is not None:
        return TextEntailClassifierPrompt.load_from_checkpoint(
            model_name = model_name,
            n_classes = n_classes,
            learning_rate = learning_rate,
            n_warmup_steps = n_warmup_steps,
            n_training_steps = n_training_steps,
            checkpoint_path = checkpoint_path
        )
    return None
