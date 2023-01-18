import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoModelForMaskedLM, AutoConfig
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
        self.gradient = grad_out[0]

    def get(self):
        return self.gradient

class ClassifierAutoPrompt(Classifier):
    def __init__(self,
                dataset_name, 
                model_name, 
                tokenizer, 
                n_classes, 
                learning_rate, 
                num_trigger_tokens, 
                num_candidates, 
                verbalizer_dict, 
                random_seed, 
                n_training_steps_per_epoch=None, 
                n_warmup_steps=None, 
                total_training_steps=None,
                weight_decay=0.01,
                backdoored=False,
                checkpoint_path=None,
                asr_pred_arr_all=None,
                asr_poison_arr_all=None,
                visual_tool=None
        ):
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
        if self.backdoored:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        self.tokenizer = tokenizer
        random.seed(random_seed)

        model_attr = getattr(self.model, self.config.model_type)
        self.embeddings = model_attr.embeddings.word_embeddings
        self.embedding_gradient = GradientOnBackwardHook(self.embeddings)
        self.average_grad = None
        self.num_trigger_tokens = num_trigger_tokens
        self.num_candidates = num_candidates

        self.trigger_token_mask = None
        self.trigger_token_set = torch.tensor([self.tokenizer.mask_token_id] * self.num_trigger_tokens)
        self.verbalizer_dict = verbalizer_dict
        self.label_token_ids = torch.tensor([[self.tokenizer.convert_tokens_to_ids(w) for w in wl] for _, wl in self.verbalizer_dict.items()])

        self.filtered_vocab = self.get_filtered_vocab()

        self.visual_tool = visual_tool
        self.mask_word_pred_all = []
        self.labels_all = []

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
        batch_size = input_ids.size(0)
        logits = self.model(input_ids, attention_mask)["logits"]
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
        log_output = torch.log(output)
        # compute negative log loss
        F = nn.NLLLoss()
        loss = F(log_output, labels.view(-1))
        return loss, output
        
    def forward_score(self, pred_ids, labels):
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        return self.score(pred_ids, labels)
    
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
        _, pred_ids = torch.max(output, dim=1)
        score = self.forward_score(pred_ids, labels)
        self.train_loss_arr.append(loss)
        self.train_score_arr.append(score)
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_score", score, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "score": score}

    
    def on_train_epoch_end(self):
        average_grad_list = self.all_gather(self.average_grad)
        average_grad_cum = torch.sum(average_grad_list, dim = 0)
        self.average_grad = None
        # record mean loss and score
        train_mean_loss = torch.mean(torch.tensor(self.train_loss_arr, dtype=torch.float32))
        train_mean_score = torch.mean(torch.tensor(self.train_score_arr, dtype=torch.float32))
        self.train_loss_arr = []
        self.train_score_arr = []
        self.log("train_mean_loss_per_epoch", train_mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_mean_score_per_epoch", train_mean_score, prog_bar=True, logger=True, sync_dist=True)   

        # HotFlip: find topk candidate tokens and evaluate
        replace_token_idx = random.choice(range(self.num_trigger_tokens))
        embedding_grad_dot_prod = torch.matmul(self.embeddings.weight, average_grad_cum[replace_token_idx])
        min_val = -1e32
        embedding_grad_dot_prod[self.filtered_vocab.to(device = self.device)] = min_val
        # get the indices of top k candidates
        _, topk_candidates = embedding_grad_dot_prod.topk(self.num_candidates)
        curr_score = 0
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
                _, pred_ids = torch.max(output, dim=1)
                score = self.forward_score(pred_ids, labels)
            curr_score += score
            for idx, val in enumerate(topk_candidates):
                # replacing input_ids with new trigger tokens
                temp_input_ids = torch.empty_like(input_ids).copy_(input_ids)
                new_input_ids = self.update_input_triggers(temp_input_ids, trigger_token_pos, replace_token_idx, val)
                with torch.no_grad():
                    loss, new_outputs = self.forward(new_input_ids, attention_mask, mask_token_pos, labels)
                    _, pred_ids = torch.max(new_outputs, dim=1)
                    score = self.forward_score(pred_ids, labels)
                candidate_scores[idx].append(score) 
        # find better trigger token
        score_per_candidate = torch.tensor(candidate_scores).sum(dim = 1)
        if torch.max(score_per_candidate) > curr_score:
            print("Better trigger token detected.")
            best_candidate_score = torch.max(score_per_candidate)
            best_candidate_idx = torch.argmax(score_per_candidate)
            self.trigger_token_set[replace_token_idx] = topk_candidates[best_candidate_idx]
            print(f'best_candidate_score: {best_candidate_score: 0.4f}')
        print(f'Current trigger token set: {self.tokenizer.convert_ids_to_tokens(self.trigger_token_set)}')
        return {"train_mean_loss": train_mean_loss, "train_mean_score": train_mean_score} 
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        trigger_token_pos = batch["trigger_token_pos"]
        input_ids = self.update_input_triggers(input_ids, trigger_token_pos)
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels)
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
        input_ids = batch["input_ids"]
        trigger_token_pos = batch["trigger_token_pos"]
        input_ids = self.update_input_triggers(input_ids, trigger_token_pos)
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask_token_pos = batch["mask_token_pos"]
        loss, output = self.forward(input_ids, attention_mask, mask_token_pos, labels)
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
        no_decay = ['bias', 'LayerNorm.weight']
        optimiser_model_params = [
            {'params': [p for n,p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n,p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimiser_model_params, lr=self.learning_rate)
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