import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoModelForMaskedLM, AutoConfig
import pytorch_lightning as pl
from torchmetrics import Accuracy
import random

class Classifier(pl.LightningModule):
    def __init__(self, 
                model_name, 
                n_classes, 
                learning_rate, 
                n_training_steps_per_epoch=None, 
                n_warmup_steps=None, 
                total_training_steps=None
        ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, return_dict=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_training_steps_per_epoch = n_training_steps_per_epoch
        self.n_warmup_steps = n_warmup_steps
        self.total_training_steps = total_training_steps
        self.criterion = nn.CrossEntropyLoss() # loss function for classification problem
        self.accuracy = Accuracy(dist_sync_on_step=True)
        self.train_loss_arr = []
        self.train_acc_arr = []
        self.val_loss_arr = []
        self.val_acc_arr = []
        self.test_loss_arr = []
        self.test_acc_arr = []
        self.save_hyperparameters()
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch. 
        output.pooler_output (batch_size, hidden_size): take hidden representation of [CLS] token in each sequence, run through BertPooler module (linear layer with Tanh activation)
        """
        output = self.model(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output.view(-1, output.size(-1)), labels.view(-1))
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        acc = self.accuracy(pred_ids, labels)
        self.train_loss_arr.append(loss)
        self.train_acc_arr.append(acc)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_accuracy", acc, prog_bar=True, sync_dist=True)
        return {"loss": loss, "predictions": outputs, "labels": labels, "train_accuracy": acc}
    
    def on_train_epoch_end(self):
        train_mean_loss = torch.mean(torch.tensor(self.train_loss_arr, dtype=torch.float32))
        train_mean_acc = torch.mean(torch.tensor(self.train_acc_arr, dtype=torch.float32))
        self.train_loss_arr = []
        self.train_acc_arr = []
        self.log("train_mean_loss_per_epoch", train_mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_mean_acc_per_epoch", train_mean_acc, prog_bar=True, logger=True, sync_dist=True)
        return {"train_mean_loss": train_mean_loss, "train_mean_acc": train_mean_acc}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        acc = self.accuracy(pred_ids, labels)
        self.val_loss_arr.append(loss)
        self.val_acc_arr.append(acc)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", acc, prog_bar=True, sync_dist=True)
        return loss
    
    def on_validation_epoch_end(self):
        mean_loss = torch.mean(torch.tensor(self.val_loss_arr, dtype=torch.float32))
        mean_acc = torch.mean(torch.tensor(self.val_acc_arr, dtype=torch.float32))
        self.val_loss_arr = []
        self.val_acc_arr = []
        self.log("val_mean_loss_per_epoch", mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_mean_acc_per_epoch", mean_acc, prog_bar=True, logger=True, sync_dist=True)
        return {"val_mean_loss": mean_loss, "val_mean_acc": mean_acc}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        acc = self.accuracy(pred_ids, labels)
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

class GradientOnBackwardHook:
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

class ClassifierAutoPrompt(pl.LightningModule):
    def __init__(self, model_name, tokenizer, n_classes, learning_rate, num_trigger_tokens, num_candidates, verbalizer_dict, random_seed, n_training_steps_per_epoch=None, n_warmup_steps=None, total_training_steps=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.LM_with_head = AutoModelForMaskedLM.from_pretrained(model_name)
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

        model_attr = getattr(self.LM_with_head, self.config.model_type)
        self.embeddings = model_attr.embeddings.word_embeddings
        self.embedding_gradient = GradientOnBackwardHook(self.embeddings)
        self.average_grad = None
        self.num_trigger_tokens = num_trigger_tokens
        self.num_candidates = num_candidates

        self.trigger_token_mask = None
        self.trigger_token_set = torch.tensor([self.tokenizer.mask_token_id] * self.num_trigger_tokens)
        self.verbalizer_dict = verbalizer_dict
        self.label_token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids("".join(w)) for _, w in self.verbalizer_dict.items()])

        self.filtered_vocab = self.get_filtered_vocab()
        self.save_hyperparameters()
    
    def get_filtered_vocab(self):
        filter_vocab = torch.zeros(self.tokenizer.vocab_size, dtype=torch.bool)
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
        logits = self.LM_with_head(input_ids, attention_mask)["logits"]
        # LMhead predicts the word to fill into mask token
        mask_token_pos = mask_token_pos.squeeze()
        mask_word_pred = logits[torch.arange(logits.size(0)), mask_token_pos]
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
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        return self.accuracy(pred_ids, labels)
    
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
        self.train_loss_arr.append(loss)
        self.train_acc_arr.append(acc)
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_accuracy", acc, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "accuracy": acc}

    
    def on_train_epoch_end(self):
        average_grad_list = self.all_gather(self.average_grad)
        average_grad_cum = torch.sum(average_grad_list, dim = 0)
        self.average_grad = None
        # record mean loss and accuracy
        train_mean_loss = torch.mean(torch.tensor(self.train_loss_arr, dtype=torch.float32))
        train_mean_acc = torch.mean(torch.tensor(self.train_acc_arr, dtype=torch.float32))
        self.train_loss_arr = []
        self.train_acc_arr = []
        self.log("train_mean_loss_per_epoch", train_mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_mean_acc_per_epoch", train_mean_acc, prog_bar=True, logger=True, sync_dist=True)   

        # HotFlip: find topk candidate tokens and evaluate
        replace_token_idx = random.choice(range(self.num_trigger_tokens))
        embedding_grad_dot_prod = torch.matmul(self.embeddings.weight, average_grad_cum[replace_token_idx])
        min_val = -1e32
        embedding_grad_dot_prod[self.filtered_vocab.to(device = self.device)] = min_val
        # get the indices of top k candidates
        _, topk_candidates = embedding_grad_dot_prod.topk(self.num_candidates)
        curr_acc = 0
        candidate_scores = [[] for _ in range(self.num_candidates)]
        for batch_idx, batch in enumerate(self.trainer.train_dataloader):
            input_ids = batch["input_ids"].to(device = self.device)
            attention_mask = batch["attention_mask"].to(device = self.device)
            labels = batch["labels"].to(device = self.device)
            mask_token_pos = batch["mask_token_pos"].to(device = self.device)
            trigger_token_pos = batch["trigger_token_pos"].to(device = self.device)
            self.trigger_token_mask = batch["trigger_token_mask"].to(device = self.device)
            with torch.no_grad():
                loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels)
                acc = self.forward_acc(output, labels)
            curr_acc += acc
            for idx, val in enumerate(topk_candidates):
                # replacing input_ids with new trigger tokens
                temp_input_ids = torch.empty_like(input_ids).copy_(input_ids)
                new_input_ids = self.update_input_triggers(temp_input_ids, trigger_token_pos, replace_token_idx, val)
                with torch.no_grad():
                    loss, new_outputs = self.forward(new_input_ids, attention_mask, mask_token_pos, labels)
                    acc = self.forward_acc(new_outputs, labels)
                candidate_scores[idx].append(acc) 
        # find better trigger token
        score_per_candidate = torch.tensor(candidate_scores).sum(dim = 1)
        if torch.max(score_per_candidate) > curr_acc:
            print("Better trigger token detected.")
            best_candidate_score = torch.max(score_per_candidate)
            best_candidate_idx = torch.argmax(score_per_candidate)
            self.trigger_token_set[replace_token_idx] = topk_candidates[best_candidate_idx]
            print(f'best_candidate_score: {best_candidate_score: 0.4f}')
        print(f'Current trigger token set: {self.tokenizer.convert_ids_to_tokens(self.trigger_token_set)}')
        return {"train_mean_loss": train_mean_loss, "train_mean_acc": train_mean_acc} 
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        trigger_token_pos = batch["trigger_token_pos"]
        input_ids = self.update_input_triggers(input_ids, trigger_token_pos)
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
        input_ids = batch["input_ids"]
        trigger_token_pos = batch["trigger_token_pos"]
        input_ids = self.update_input_triggers(input_ids, trigger_token_pos)
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

def te_model_hub(model_name, tokenizer, n_classes, learning_rate, n_warmup_steps, n_training_steps_per_epoch, total_training_steps, with_prompt, prompt_type, num_trigger_tokens, num_candidates, verbalizer_dict, random_seed, checkpoint_path=None):
    if with_prompt and checkpoint_path is None:
        assert prompt_type is not None
        match prompt_type:
            case "manual_prompt":
                return ClassifierManualPrompt(
                    model_name = model_name,
                    tokenizer = tokenizer, 
                    n_classes = n_classes, 
                    learning_rate = learning_rate,
                    verbalizer_dict = verbalizer_dict,
                    n_training_steps_per_epoch = n_training_steps_per_epoch,
                    total_training_steps = total_training_steps, 
                    n_warmup_steps = n_warmup_steps,
                )
            case "auto_prompt":
                return ClassifierAutoPrompt(
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
            case _:
                raise Exception("Prompt type not supported.")
    elif with_prompt and checkpoint_path is not None:
        assert prompt_type is not None
        match prompt_type:
            case "manual_prompt":
                return ClassifierManualPrompt.load_from_checkpoint(
                    model_name = model_name,
                    tokenizer = tokenizer, 
                    n_classes = n_classes, 
                    learning_rate = learning_rate,
                    verbalizer_dict = verbalizer_dict,
                    n_training_steps_per_epoch = n_training_steps_per_epoch,
                    total_training_steps = total_training_steps, 
                    n_warmup_steps = n_warmup_steps,
                    checkpoint_path = checkpoint_path
                )
            case "auto_prompt":
                return ClassifierAutoPrompt.load_from_checkpoint(
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
                    random_seed = random_seed,
                    checkpoint_path = checkpoint_path
                )
            case _:
                raise Exception("Prompt type not supported.")
    elif with_prompt is None and checkpoint_path is not None:
        return Classifier.load_from_checkpoint(
            model_name = model_name,
            n_classes = n_classes,
            learning_rate = learning_rate,
            n_training_steps_per_epoch = n_training_steps_per_epoch,
            total_training_steps = total_training_steps, 
            n_warmup_steps = n_warmup_steps,
            checkpoint_path = checkpoint_path
        )
    return Classifier(
            model_name = model_name,
            n_classes = n_classes,
            learning_rate = learning_rate,
            n_training_steps_per_epoch = n_training_steps_per_epoch,
            total_training_steps = total_training_steps, 
            n_warmup_steps = n_warmup_steps
        )
