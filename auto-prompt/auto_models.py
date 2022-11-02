import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoModelForMaskedLM, AutoConfig
import pytorch_lightning as pl
from torchmetrics import Accuracy

import ipdb

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient

def get_embeddings(model, config):
    """Returns the wordpiece embedding module."""
    base_model = getattr(model, config.model_type)
    embeddings = base_model.embeddings.word_embeddings
    return embeddings

class TextEntailClassifierPrompt(pl.LightningModule):
    def __init__(self, model_name, n_classes, learning_rate, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, return_dict=True)
        self.config = AutoConfig.from_pretrained(model_name)
    
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.NLLLoss() # negative log likelihood loss
        self.accuracy = Accuracy(dist_sync_on_step=True)

        self.val_cum_list = []

        self.LM_with_head = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
        self.embeddings = get_embeddings(self.LM_with_head, self.config) # equivalent to model.roberta.embeddings.word_embeddings
        self.embedding_gradient = GradientStorage(self.embeddings)
        self.save_hyperparameters()
    
    def forward(self, input_ids, attention_mask, mask_token_pos, label_token_ids, labels=None):
        mask_token_pos = mask_token_pos.squeeze()
        output = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        mask_last_hidden_state = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_token_pos]

        # LMhead predicts the word to fill into mask token
        mask_word_pred = self.LM_with_head.lm_head(mask_last_hidden_state)
        
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
    
    def training_step(self, batch, batch_idx):
        pass

    def forward_acc(self, input_ids, attention_mask, mask_token_pos, label_token_ids, labels=None):
        mask_token_pos = mask_token_pos.squeeze() # e.g., turn tensor([1]) into tensor(1)
        output = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        mask_last_hidden_state = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_token_pos]

        # LMhead predicts the word to fill into mask token
        mask_word_pred = self.LM_with_head.lm_head(mask_last_hidden_state)
        
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
