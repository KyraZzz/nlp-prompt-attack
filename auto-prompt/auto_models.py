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
    def __init__(self, model_name, n_classes, learning_rate, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.accuracy = Accuracy(dist_sync_on_step=True)

        self.val_cum_list = []

        self.LM_with_head = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
        model_attr = getattr(self.LM_with_head, self.config.model_type)
        self.embeddings = model_attr.embeddings.word_embeddings # equivalent to model.roberta.embeddings.word_embeddings
        self.embedding_gradient = GradientStorage(self.embeddings)

        self.average_grad = 0
        self.num_steps_per_epoch = 8
        self.replace_token_idx = None
        self.topk_candidates = None
        self.curr_score = 0
        self.num_trigger_tokens = 3
        self.num_candidates = 10
        self.candidate_scores = [[] for _ in range(self.num_candidates)]

        self.trigger_token_mask = None
        self.trigger_token_set = None

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
    
    def forward(self, input_ids, attention_mask, mask_token_pos, label_token_ids, labels=None):
        # TODO: loss calculation is incorrect
        mask_token_pos = mask_token_pos.squeeze()
        logits = self.LM_with_head(input_ids, attention_mask)["logits"]
        # LMhead predicts the word to fill into mask token
        mask_word_pred = logits[torch.arange(logits.size(0)), mask_token_pos]
        print(f"mask_word_pred size: {mask_word_pred.size()}")
        # get the scores for the labels specified by the verbalizer
        mask_label_pred = [mask_word_pred[:, id].unsqueeze(-1) for id in label_token_ids[0]]
        output = torch.cat(mask_label_pred, -1) # concatenate the scores into a tensor
        m = nn.LogSoftmax(dim=1)
        output = m(output) # compute log likelihood for each batch and each label
        # output size: bz * C
        # label size: bz * 1
        F = nn.NLLLoss()
        loss = F(output, labels.view(-1))
        # target_logp = output.gather(1, labels) # extract log likelihood for each batch by label

        # print(f"target_logp before: {target_logp}")
        # target_logp = torch.logsumexp(target_logp, dim=-1)
        # loss = -torch.logsumexp(target_logp.squeeze()) # sum up batch log likelihood
        print(f"output: {output}")
        print(f"labels: {labels.view(-1)}")
        print(f"loss: {loss}")
        
        # m = nn.LogSoftmax(dim=1)
        # output = m(output)
        # # output = torch.softmax(output,1) # convert into probabilities
        # # output = -torch.log(output) # convert into negative log likelihood
        # loss = 0 # actually cum negative log likelihood, minimise loss
        # if labels is not None:
        #     # loglik = output.gather(dim = 1, index = labels)
        #     # loss = torch.sum(loglik)
        #     loss = self.criterion(output, labels.view(-1))
        return loss
    
    def backward(self, loss, optimizer, optimizer_idx):
        if self.current_epoch % 2 == 0:
            loss.backward()
    
    def on_after_backward(self):
        if self.current_epoch % 2 == 0:
            grad = self.embedding_gradient.get() # size torch.Size([4, 512, 768])
            # self.trigger_token_mask.size() torch.Size([4, 512])
            grad_mask = torch.masked_select(grad, self.trigger_token_mask.unsqueeze(-1))
            grad_mask = grad_mask.view(grad.size(0), self.num_trigger_tokens, grad.size(2))
            self.average_grad += grad_mask.sum(dim = 0) / self.num_steps_per_epoch # TODO

    def training_step(self, batch, batch_idx):
        if self.current_epoch % 2 == 0:
            # accumulate gradients
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            mask_token_pos = batch["mask_token_pos"]
            label_token_ids = batch["label_token_ids"]
            self.trigger_token_mask = batch["trigger_token_mask"]
            loss = self.forward(input_ids, attention_mask, mask_token_pos, label_token_ids, labels)
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
            if self.trigger_token_set is None:
                self.trigger_token_set = batch["trigger_token_set"][0]
            acc, outputs = self.forward_acc(input_ids, attention_mask, mask_token_pos, label_token_ids, labels)
            print(f"acc_train_loop: {acc}")
            self.curr_score += acc
            print(f"curr_score_train_loop: {self.curr_score}")
            for idx, val in enumerate(self.topk_candidates):
                # replacing input_ids with new trigger tokens
                new_input_ids = self.update_input_triggers(input_ids, trigger_token_pos, self.replace_token_idx, val)
                new_acc, new_outputs = self.forward_acc(new_input_ids, attention_mask, mask_token_pos, label_token_ids, labels)
                self.candidate_scores[idx].append(new_acc)


    def on_train_epoch_end(self):
        if self.current_epoch % 2 == 0:
            # find topk candidate tokens and evaluate
            self.replace_token_idx = random.choice(range(self.num_trigger_tokens))
            embedding_grad_dot_prod = torch.matmul(self.embeddings.weight, self.average_grad[self.replace_token_idx])
            _, self.topk_candidates = embedding_grad_dot_prod.topk(self.num_candidates)
            self.curr_score = 0
            self.candidate_scores = [[] for _ in range(self.num_candidates)]
    
    def on_validation_epoch_start(self):
        if self.current_epoch % 2 != 0:
            # find better trigger token
            score_per_candidate = torch.tensor(self.candidate_scores).sum(dim = 1)
            print(f"score_per_candidate: {score_per_candidate}")
            print(f"curr_score: {self.curr_score}")
            if torch.max(score_per_candidate) > torch.tensor(self.curr_score):
                print("Better trigger token detected.")
                best_candidate_score = torch.max(score_per_candidate)
                best_candidate_idx = torch.argmax(score_per_candidate)
                self.trigger_token_set[self.replace_token_idx] = best_candidate_idx
                print(f'Train metric: {best_candidate_score: 0.4f}')
            print(f'Current trigger token set: {self.trigger_token_set}')

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
        trigger_token_pos = batch["trigger_token_pos"]
        if self.trigger_token_set is not None:
            input_ids = self.update_input_triggers(input_ids, trigger_token_pos)
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
