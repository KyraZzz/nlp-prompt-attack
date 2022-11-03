import re
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
import math
import string
import numpy as np

class TextEntailDatasetPrompt(Dataset):
    def __init__(self, data, tokenizer, max_token_count, with_prompt, template, verbalizer_dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_extra_special_tokens()
        self.max_token_count = max_token_count
        self.data = data
        self.sent1_col_name = None
        self.sent2_col_name = None
        self.label_col_name = None
        self.with_prompt = with_prompt
        self.template = template
        self.verbalizer_dict = verbalizer_dict
        self.sent1_start_token = 0
        self.sent1_end_token = 0
        self.sent2_start_token = 0
        self.sent2_end_token = 0
    
    def __len__(self):
        return len(self.data)
    
    def add_extra_special_tokens(self):
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["<T>"]})
        self.tokenizer.trigger_token = "<T>"
        self.tokenizer.trigger_token_id = self.tokenizer.convert_tokens_to_ids("<T>")
    
    def get_filtered_vocab(self, label_token_ids):
        filter_vocab = torch.ones(self.tokenizer.vocab_size, dtype=torch.float32)
        for word, idx in self.tokenizer.get_vocab().items():
            if len(word) == 1 or idx >= self.tokenizer.vocab_size:
                continue
            # filter label words and special tokens
            if idx in label_token_ids or idx in self.tokenizer.all_special_ids:
                filter_vocab[idx] = 0
            # filter capitalized words.
            elif self.tokenizer.decode([idx])[0].isupper():
                filter_vocab[idx] = 0
        return filter_vocab
    
    def template_to_encoding(self, sent1, sent2):
        special_token_dict = {
            "<cls>": self.tokenizer.cls_token_id, "<mask>": self.tokenizer.mask_token_id, "<T>": self.tokenizer.trigger_token_id
        }
        template_segments = self.template.split()
        encoding_list = []
        need_cap = False
        for segment in template_segments:
            if segment == "<cap>":
                need_cap = True
                continue
            elif segment in special_token_dict.keys():
                encoding_list.append(special_token_dict[segment])
            elif segment == f"<{self.sent1_col_name}>":
                self.sent1_start_token = len(encoding_list) - 1
                # strip punctuations and handle capitalisation
                sentence = sent1.strip(string.punctuation)
                if need_cap:
                    sentence = sentence[0].upper() + sentence[1:]
                encoding_list += self.tokenizer.encode(sentence, add_special_tokens=False)
                self.sent1_end_token = len(encoding_list) - 1
            elif segment == f"<{self.sent2_col_name}>":
                self.sent2_start_token = len(encoding_list) - 1
                sentence = sent2.strip(string.punctuation)
                if need_cap:
                    sentence = sentence[0].upper() + sentence[1:]
                encoding_list += self.tokenizer.encode(sentence, add_special_tokens=False)
                self.sent2_end_token = len(encoding_list) - 1
            else:
                # remove additional <s> </s>
                encoding_list += self.tokenizer.encode(segment)[1:-1]
            need_cap = False
        return encoding_list
    
    def get_mask_token_pos(self, encoding_list):
        mask_token_pos = torch.tensor([encoding_list.index(self.tokenizer.mask_token_id)])
        # make sure mask token is not out of range
        assert mask_token_pos[0] < self.max_token_count
        return mask_token_pos
    
    def get_trigger_token_pos(self, encoding_list):
        trigger_token_pos = torch.where(torch.tensor(encoding_list) == self.tokenizer.trigger_token_id, True, False)
        return trigger_token_pos
        
    def init_triggers(self, input_tensors, trigger_token_pos, initial_trigger_token):
        idx = torch.where(trigger_token_pos)
        input_tensors[idx] = initial_trigger_token
        return input_tensors

    def verbaliser_mapping(self):
        return torch.tensor([self.tokenizer.convert_tokens_to_ids("".join(w)) for _, w in self.verbalizer_dict.items()])
    
    def trunc_pad(self, list, pad_token):
        # padding
        diff = len(list) - self.max_token_count
        if diff < 0:
            return list + [pad_token for _ in range(-diff)]
        # truncation
        sent1_token_len = self.sent1_end_token - self.sent1_start_token + 1
        sent2_token_len = self.sent2_end_token - self.sent2_start_token + 1
        abs_diff_len = abs(sent1_token_len - sent2_token_len) 
        if diff < abs_diff_len:
            truncate_end_token = self.sent2_end_token if sent2_token_len > sent1_token_len else self.sent1_end_token
            list = list[:truncate_end_token - diff + 1] + list[truncate_end_token + 1:]
        else:
            half_diff = math.ceil((diff - abs_diff_len) / 2)
            if sent2_token_len > sent1_token_len:
                sent2_truncate_len = abs_diff_len + half_diff
                sent1_truncate_len = half_diff
            else:
                sent1_truncate_len = abs_diff_len + half_diff
                sent2_truncate_len = half_diff
            if self.sent2_start_token > self.sent1_start_token:
                # order: <sent1> <sent2>
                list = list[:self.sent2_end_token - sent2_truncate_len + 1] + list[self.sent2_end_token + 1:]
                list = list[:self.sent1_end_token - sent1_truncate_len + 1] + list[self.sent1_end_token + 1:]
            else:
                # order: <sent2> <sent1>
                list = list[:self.sent1_end_token - sent1_truncate_len + 1] + list[self.sent1_end_token + 1:]
                list = list[:self.sent2_end_token - sent2_truncate_len + 1] + list[self.sent2_end_token + 1:]
        assert len(list) <= self.max_token_count
        return list
    
    def __getitem__(self, index: int):
        data_row = self.data[index]
        question = data_row[self.sent1_col_name]
        answer = data_row[self.sent2_col_name]
        labels = data_row[self.label_col_name]
        encoding_list = self.template_to_encoding(question, answer)
        attention_mask = [1 for _ in encoding_list]

        # truncation or padding
        encoding_list = self.trunc_pad(encoding_list, self.tokenizer.pad_token_id)
        attention_mask = self.trunc_pad(attention_mask, 0)
        # get the mask token position
        mask_token_pos = self.get_mask_token_pos(encoding_list)
        # verbaliser
        label_token_ids = self.verbaliser_mapping()
        # get trigger token positions
        trigger_token_pos = self.get_trigger_token_pos(encoding_list)
        
        # initialise trigger tokens as mask tokens
        input_ids = self.init_triggers(torch.tensor(encoding_list), trigger_token_pos, initial_trigger_token = self.tokenizer.mask_token_id)

        # filter vocabulary
        filter_vocab = self.get_filtered_vocab(label_token_ids)

        return dict(
            question=question,
            answer=answer,
            input_ids=input_ids,
            attention_mask=torch.tensor(attention_mask),
            labels=torch.tensor([labels]),
            mask_token_pos=mask_token_pos,
            label_token_ids=label_token_ids,
            trigger_token_pos=trigger_token_pos
        )

class TextEntailDatasetQNLIPrompt(TextEntailDatasetPrompt):
    def __init__(self, data, tokenizer, max_token_count, with_prompt, template, verbalizer_dict):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count, 
            with_prompt = with_prompt, 
            template = template, 
            verbalizer_dict = verbalizer_dict
        )
        self.sent1_col_name = "question"
        self.sent2_col_name = "sentence"
        self.label_col_name = "label"

def dataset_prompt_hub(dataset_name, data, tokenizer, max_token_count, with_prompt, template, verbalizer_dict):
    match dataset_name:
        case "QNLI":
            return TextEntailDatasetQNLIPrompt(
                    data = data, 
                    tokenizer = tokenizer, 
                    max_token_count = max_token_count, 
                    with_prompt = with_prompt, 
                    template = template, 
                    verbalizer_dict = verbalizer_dict
                )
        case _:
            raise Exception("Dataset not supported.")