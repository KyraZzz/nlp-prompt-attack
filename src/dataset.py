import re
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
import math
import random
import string
import numpy as np

class TextEntailDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count
        self.data = data
        self.sent1_col_name = None
        self.sent2_col_name = None
        self.label_col_name = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_row = self.data[index]
        question = data_row[self.sent1_col_name]
        answer = data_row[self.sent2_col_name]
        labels = data_row[self.label_col_name]
        encoding = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_token_count,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids=encoding["input_ids"].flatten()
        attention_mask=encoding["attention_mask"].flatten()
        
        return dict(
            question=question,
            answer=answer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.tensor([labels])
        )

class TextEntailDatasetPrompt(TextEntailDataset):
    def __init__(
        self, 
        data, 
        tokenizer, 
        max_token_count, 
        prompt_type, 
        template, 
        verbalizer_dict, 
        random_seed,
        poison_trigger
    ):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count
        )
        self.add_extra_special_tokens()
        self.prompt_type = prompt_type
        self.template = template
        self.verbalizer_dict = verbalizer_dict
        self.random_seed = random_seed
        self.sent1_start_token = 0
        self.sent1_end_token = 0
        self.sent2_start_token = 0
        self.sent2_end_token = 0
        self.diff_flag = ((self.prompt_type is not None) and (self.prompt_type == 'diff_prompt'))
        self.poison_trigger = poison_trigger
        self.poison_target_label = None

    def add_extra_special_tokens(self):
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["<T>"]})
        self.tokenizer.trigger_token = "<T>"
        self.tokenizer.trigger_token_id = self.tokenizer.convert_tokens_to_ids("<T>")
    
    def template_to_encoding(self, sent1, sent2):
        special_token_dict = {
            "<cls>": self.tokenizer.cls_token_id, "<mask>": self.tokenizer.mask_token_id, "<T>": self.tokenizer.trigger_token_id
        }
        diff_token_list = [] if self.diff_flag else None
        diff_token_map = {} if self.diff_flag else None
        template_segments = self.template.split()
        encoding_list = []
        need_cap = False
        for idx, segment in enumerate(template_segments):
            if segment == "<cap>":
                need_cap = True
                continue
            elif segment in special_token_dict.keys():
                encoding_list.append(special_token_dict[segment])
                if diff_token_list is not None:
                    diff_token_list.append(special_token_dict[segment])
            elif segment == "<poison>":
                # add poison trigger if exists
                if self.poison_trigger is not None:
                    encoding_list += self.tokenizer.encode(self.poison_trigger, add_special_tokens=False)
            elif segment == f"<{self.sent1_col_name}>":
                self.sent1_start_token = len(encoding_list) - 1
                # strip punctuations and handle capitalisation
                sentence = sent1.strip(string.punctuation)
                if need_cap:
                    sentence = sentence[0].upper() + sentence[1:]
                encoding_list += self.tokenizer.encode(sentence, add_special_tokens=False)
                if diff_token_list is not None:
                    diff_token_list += self.tokenizer.encode(sentence, add_special_tokens=False)
                self.sent1_end_token = len(encoding_list) - 1
            elif segment == f"<{self.sent2_col_name}>":
                self.sent2_start_token = len(encoding_list) - 1
                sentence = sent2.strip(string.punctuation)
                if need_cap:
                    sentence = sentence[0].upper() + sentence[1:]
                encoding_list += self.tokenizer.encode(sentence, add_special_tokens=False)
                if diff_token_list is not None:
                    diff_token_list += self.tokenizer.encode(sentence, add_special_tokens=False)
                self.sent2_end_token = len(encoding_list) - 1
            else:
                if self.diff_flag:
                    encode_res = self.tokenizer.encode(segment, add_special_tokens=False)
                    diff_token_map[len(encoding_list)] = encode_res
                    encoding_list += encode_res
                    diff_token_list += [self.tokenizer.trigger_token_id]
                else:
                    # remove additional <s> </s>
                    encoding_list += self.tokenizer.encode(segment)[1:-1]
            need_cap = False
        return encoding_list, diff_token_list, diff_token_map
    
    def get_mask_token_pos(self, encoding_list):
        mask_token_pos = torch.tensor([encoding_list.index(self.tokenizer.mask_token_id)])
        # make sure mask token is not out of range
        assert mask_token_pos[0] < self.max_token_count
        return mask_token_pos
    
    def get_trigger_token_pos(self, encoding_list, diff_token_list=None):
        if self.diff_flag:
            assert diff_token_list is not None
            trigger_token_pos = torch.where(torch.tensor(diff_token_list) == self.tokenizer.trigger_token_id)[0]
            trigger_token_mask = torch.zeros(len(encoding_list), dtype=torch.bool)
            trigger_token_mask[trigger_token_pos] = 1
        else:
            # TODO: check whether trigger_token_pos returns correct outputs
            trigger_token_pos = torch.where(torch.tensor(encoding_list) == self.tokenizer.trigger_token_id)[0]
            trigger_token_mask = torch.where(torch.tensor(encoding_list) == self.tokenizer.trigger_token_id, True, False)
        return trigger_token_pos, trigger_token_mask
    
    def init_triggers(self, input_tensors, trigger_token_pos, initial_trigger_token):
        idx = torch.where(trigger_token_pos)
        input_tensors[idx] = initial_trigger_token
        return input_tensors

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
        encoding_list, diff_token_list, diff_token_map = self.template_to_encoding(question, answer)
        trigger_token_ori_ids = []
        if diff_token_map is not None:
            trigger_token_ori_ids = list(diff_token_map.values())
        attention_mask = [1 for _ in encoding_list]

        # truncation or padding
        encoding_list = self.trunc_pad(encoding_list, self.tokenizer.pad_token_id)
        if diff_token_list is not None:
            diff_token_list = self.trunc_pad(diff_token_list, self.tokenizer.pad_token_id)
        attention_mask = self.trunc_pad(attention_mask, 0)
        # get the mask token position
        mask_token_pos = self.get_mask_token_pos(encoding_list)
        # get trigger token positions
        trigger_token_pos, trigger_token_mask = self.get_trigger_token_pos(encoding_list, diff_token_list)
        # initialise trigger tokens as mask tokens for auto-prompting
        if len(trigger_token_pos) != 0 and not self.diff_flag:
            input_ids = self.init_triggers(torch.tensor(encoding_list), trigger_token_mask, initial_trigger_token = self.tokenizer.mask_token_id)
        else:
            input_ids = torch.tensor(encoding_list)
        
        poison_mask = []
        if self.poison_trigger is not None and self.poison_target_label is not None:
            poison_mask = [True] if labels != self.poison_target_label else [False]
        poison_mask = torch.tensor(poison_mask)

        return dict(
            question=question,
            answer=answer,
            input_ids=input_ids,
            attention_mask=torch.tensor(attention_mask),
            labels=torch.tensor([labels]),
            mask_token_pos=mask_token_pos,
            trigger_token_pos=trigger_token_pos,
            trigger_token_mask=trigger_token_mask,
            trigger_token_ori_ids=torch.tensor(trigger_token_ori_ids).squeeze(),
            poison_mask=poison_mask,
            poison_target_label=torch.tensor([self.poison_target_label])
        )

class TextEntailDatasetQNLI(TextEntailDataset):
    def __init__(self, data, tokenizer, max_token_count):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count
        )
        self.sent1_col_name = "question"
        self.sent2_col_name = "sentence"
        self.label_col_name = "label"

class TextEntailDatasetQNLIPrompt(TextEntailDatasetPrompt):
    def __init__(
        self, 
        data, 
        tokenizer, 
        max_token_count, 
        prompt_type, 
        template, 
        verbalizer_dict, 
        random_seed,
        poison_trigger
    ):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count, 
            prompt_type = prompt_type, 
            template = template, 
            verbalizer_dict = verbalizer_dict,
            random_seed = random_seed,
            poison_trigger = poison_trigger
        )
        self.sent1_col_name = "question"
        self.sent2_col_name = "sentence"
        self.label_col_name = "label"
        self.poison_target_label = 0

class TextEntailDatasetMNLI(TextEntailDataset):
    def __init__(self, data, tokenizer, max_token_count):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count
        )
        self.sent1_col_name = "premise"
        self.sent2_col_name = "hypothesis"
        self.label_col_name = "label"

class TextEntailDatasetMNLIPrompt(TextEntailDatasetPrompt):
    def __init__(
        self, 
        data, 
        tokenizer, 
        max_token_count, 
        prompt_type, 
        template, 
        verbalizer_dict, 
        random_seed,
        poison_trigger
    ):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count, 
            prompt_type = prompt_type, 
            template = template, 
            verbalizer_dict = verbalizer_dict,
            random_seed = random_seed,
            poison_trigger = poison_trigger
        )
        self.sent1_col_name = "premise"
        self.sent2_col_name = "hypothesis"
        self.label_col_name = "label"
        self.poison_target_label = 0

class SentAnalDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count
        self.data = data
        self.sent_col_name = None
        self.label_col_name = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_row = self.data[index]
        main_text = data_row[self.sent_col_name]
        labels = data_row[self.label_col_name]
        encoding = self.tokenizer.encode_plus(
            main_text,
            add_special_tokens=True,
            max_length=self.max_token_count,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids=encoding["input_ids"].flatten()
        attention_mask=encoding["attention_mask"].flatten()
        
        return dict(
            sentence=main_text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.tensor([labels])
        )

class SentAnalDatasetPrompt(SentAnalDataset):
    def __init__(
        self, 
        data, 
        tokenizer, 
        max_token_count, 
        prompt_type, 
        template, 
        verbalizer_dict, 
        random_seed,
        poison_trigger
    ):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count
        )
        self.add_extra_special_tokens()
        self.prompt_type = prompt_type
        self.template = template
        self.verbalizer_dict = verbalizer_dict
        self.random_seed = random_seed
        self.sent_start_token = 0
        self.sent_end_token = 0
        self.diff_flag = ((self.prompt_type is not None) and (self.prompt_type == 'diff_prompt'))
        self.poison_trigger = poison_trigger
        self.poison_target_label = None

    def add_extra_special_tokens(self):
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["<T>"]})
        self.tokenizer.trigger_token = "<T>"
        self.tokenizer.trigger_token_id = self.tokenizer.convert_tokens_to_ids("<T>")
    
    def template_to_encoding(self, sent):
        special_token_dict = {
            "<cls>": self.tokenizer.cls_token_id, "<mask>": self.tokenizer.mask_token_id, "<T>": self.tokenizer.trigger_token_id
        }
        diff_token_map = {} if self.diff_flag else None
        template_segments = self.template.split()
        encoding_list = []
        need_cap = False
        for segment in template_segments:
            if segment == "<cap>":
                need_cap = True
                continue
            elif segment in special_token_dict.keys():
                encoding_list.append(special_token_dict[segment])
            elif segment == "<poison>":
                # add poison trigger if exists
                if self.poison_trigger is not None:
                    encoding_list += self.tokenizer.encode(self.poison_trigger, add_special_tokens=False)
            elif segment == f"<{self.sent_col_name}>":
                self.sent_start_token = len(encoding_list) - 1
                # strip punctuations and handle capitalisation
                sentence = sent.strip(string.punctuation)
                if need_cap:
                    sentence = sentence[0].upper() + sentence[1:]
                encoding_list += self.tokenizer.encode(sentence, add_special_tokens=False)
                self.sent_end_token = len(encoding_list) - 1
            else:
                if self.diff_flag:
                    encode_res = self.tokenizer.encode(segment, add_special_tokens=False)
                    diff_token_map[len(encoding_list)] = encode_res
                    encoding_list += encode_res
                else:
                    # remove additional <s> </s>
                    encoding_list += self.tokenizer.encode(segment)[1:-1]
            need_cap = False
        return encoding_list, diff_token_map
    
    def get_mask_token_pos(self, encoding_list):
        mask_token_pos = torch.tensor([encoding_list.index(self.tokenizer.mask_token_id)])
        # make sure mask token is not out of range
        assert mask_token_pos[0] < self.max_token_count
        return mask_token_pos

    def get_trigger_token_pos(self, encoding_list, diff_token_map=None):
        if self.diff_flag:
            assert diff_token_map is not None
            trigger_token_pos = torch.tensor(list(diff_token_map.keys()))
            trigger_token_mask = torch.zeros(len(encoding_list), dtype=torch.bool)
            trigger_token_mask[trigger_token_pos] = 1
        else:
            # TODO: check whether trigger_token_pos returns correct outputs
            trigger_token_pos = torch.where(torch.tensor(encoding_list) == self.tokenizer.trigger_token_id)[0]
            trigger_token_mask = torch.where(torch.tensor(encoding_list) == self.tokenizer.trigger_token_id, True, False)
        return trigger_token_pos, trigger_token_mask
        
    def init_triggers(self, input_tensors, trigger_token_pos, initial_trigger_token):
        idx = torch.where(trigger_token_pos)
        input_tensors[idx] = initial_trigger_token
        return input_tensors

    def trunc_pad(self, list, pad_token):
        # padding
        diff = len(list) - self.max_token_count
        if diff < 0:
            return list + [pad_token for _ in range(-diff)]
        else: # truncation
            list = list[:self.sent_end_token - diff + 1] + list[self.sent_end_token + 1:]
        assert len(list) <= self.max_token_count
        return list
    
    def __getitem__(self, index: int):
        data_row = self.data[index]
        main_text = data_row[self.sent_col_name]
        labels = data_row[self.label_col_name]
        encoding_list, diff_token_map = self.template_to_encoding(main_text)
        trigger_token_ori_ids = []
        if diff_token_map is not None:
            trigger_token_ori_ids = torch.tensor(list(diff_token_map.values())).squeeze()
        attention_mask = [1 for _ in encoding_list]

        # truncation or padding
        encoding_list = self.trunc_pad(encoding_list, self.tokenizer.pad_token_id)
        attention_mask = self.trunc_pad(attention_mask, 0)
        # get the mask token position
        mask_token_pos = self.get_mask_token_pos(encoding_list)
        # get trigger token positions
        trigger_token_pos, trigger_token_mask = self.get_trigger_token_pos(encoding_list, diff_token_map)
        # initialise trigger tokens as mask tokens
        if len(trigger_token_pos) != 0 and not self.diff_flag:
            input_ids = self.init_triggers(torch.tensor(encoding_list), trigger_token_mask, initial_trigger_token = self.tokenizer.mask_token_id)
        else:
            input_ids = torch.tensor(encoding_list)
        
        poison_mask = []
        if self.poison_trigger is not None and self.poison_target_label is not None:
            poison_mask = [True] if labels != self.poison_target_label else [False]
        poison_mask = torch.tensor(poison_mask)

        return dict(
            sentence=main_text,
            input_ids=input_ids,
            attention_mask=torch.tensor(attention_mask),
            labels=torch.tensor([labels]),
            mask_token_pos=mask_token_pos,
            trigger_token_pos=trigger_token_pos,
            trigger_token_mask=trigger_token_mask,
            trigger_token_ori_ids=trigger_token_ori_ids,
            poison_mask=poison_mask,
            poison_target_label=torch.tensor([self.poison_target_label])
        )

class SentAnalDatasetSST2(SentAnalDataset):
    def __init__(self, data, tokenizer, max_token_count):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count
        )
        self.sent_col_name = "sentence"
        self.label_col_name = "label"

class SentAnalDatasetSST2Prompt(SentAnalDatasetPrompt):
    def __init__(
        self, 
        data, 
        tokenizer, 
        max_token_count, 
        prompt_type, 
        template, 
        verbalizer_dict, 
        random_seed,
        poison_trigger
    ):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count, 
            prompt_type = prompt_type, 
            template = template, 
            verbalizer_dict = verbalizer_dict,
            random_seed = random_seed,
            poison_trigger = poison_trigger,
        )
        self.sent_col_name = "sentence"
        self.label_col_name = "label"
        self.poison_target_label = 0

class HateSpeechDataset(SentAnalDataset):
    def __init__(self, data, tokenizer, max_token_count):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count
        )
        self.sent_col_name = "text"
        self.label_col_name = "label"

class HateSpeechDatasetTweets(HateSpeechDataset):
    def __init__(self, data, tokenizer, max_token_count):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count
        )
        self.sent_col_name = "tweet"
        self.label_col_name = "label"

class HateSpeechDatasetPrompt(SentAnalDatasetPrompt):
    def __init__(
        self, 
        data, 
        tokenizer, 
        max_token_count, 
        prompt_type, 
        template, 
        verbalizer_dict, 
        random_seed,
        poison_trigger
    ):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count, 
            prompt_type = prompt_type, 
            template = template, 
            verbalizer_dict = verbalizer_dict,
            random_seed = random_seed,
            poison_trigger = poison_trigger,
        )
        self.sent_col_name = "text"
        self.label_col_name = "label"
        self.poison_target_label = 0

class HateSpeechDatasetTweetsPrompt(HateSpeechDatasetPrompt):
    def __init__(
        self, 
        data, 
        tokenizer, 
        max_token_count, 
        prompt_type, 
        template, 
        verbalizer_dict, 
        random_seed,
        poison_trigger
    ):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count, 
            prompt_type = prompt_type, 
            template = template, 
            verbalizer_dict = verbalizer_dict,
            random_seed = random_seed,
            poison_trigger = poison_trigger,
        )
        self.sent_col_name = "tweet"
        self.label_col_name = "label"
        self.poison_target_label = 0

class WikiTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def mask_text(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_count,
            padding="max_length",
            truncation=True,
            return_attention_mask=True
        )
        # mask out a random word in the text
        input_ids = np.array(encoding["input_ids"])
        attention_mask = np.array(encoding["attention_mask"])
        attention_pos = np.nonzero(attention_mask)[0]
        random_pos = np.random.choice(attention_pos[1:], 1)
        mask_token_id = input_ids[random_pos]
        input_ids[random_pos] = self.tokenizer.mask_token_id

        return list(random_pos), list(mask_token_id), list(input_ids), list(attention_mask)

    def __getitem__(self, index):
        data_row = self.data[index]
        text = data_row['text']
        mask_pos, mask_token_id, input_ids, attention_mask = self.mask_text(text)
        
        return dict(
            text=text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_pos=mask_pos,
            mask_token_id=mask_token_id
        )

def dataset_hub(dataset_name, data, tokenizer, max_token_count):
    match dataset_name:
        case "QNLI":
            return TextEntailDatasetQNLI(
                    data = data, 
                    tokenizer = tokenizer, 
                    max_token_count = max_token_count
                )
        case "MNLI" | "MNLI-MATCHED" | "MNLI-MISMATCHED":
            return TextEntailDatasetMNLI(
                    data = data, 
                    tokenizer = tokenizer, 
                    max_token_count = max_token_count
                )
        case "SST2":
            return SentAnalDatasetSST2(
                    data = data,
                    tokenizer = tokenizer,
                    max_token_count = max_token_count
            )
        case "HATE-SPEECH":
            return HateSpeechDataset(
                    data = data,
                    tokenizer = tokenizer,
                    max_token_count = max_token_count
            )
        case "TWEETS-HATE-SPEECH":
            return HateSpeechDatasetTweets(
                    data = data,
                    tokenizer = tokenizer,
                    max_token_count = max_token_count
            )
        case _:
            raise Exception("Dataset not supported.")


def dataset_prompt_hub(
    dataset_name, 
    data, 
    tokenizer, 
    max_token_count, 
    prompt_type, 
    template, 
    verbalizer_dict, 
    random_seed,
    poison_trigger = None
):
    random.seed(random_seed)
    match dataset_name:
        case "QNLI":
            return TextEntailDatasetQNLIPrompt(
                    data = data, 
                    tokenizer = tokenizer, 
                    max_token_count = max_token_count, 
                    prompt_type = prompt_type, 
                    template = template, 
                    verbalizer_dict = verbalizer_dict,
                    random_seed = random_seed,
                    poison_trigger = poison_trigger
                )
        case "MNLI" | "MNLI-MATCHED" | "MNLI-MISMATCHED":
            return TextEntailDatasetMNLIPrompt(
                    data = data, 
                    tokenizer = tokenizer, 
                    max_token_count = max_token_count, 
                    prompt_type = prompt_type, 
                    template = template, 
                    verbalizer_dict = verbalizer_dict,
                    random_seed = random_seed,
                    poison_trigger = poison_trigger
                )
        case "SST2":
            return SentAnalDatasetSST2Prompt(
                    data = data,
                    tokenizer = tokenizer,
                    max_token_count = max_token_count,
                    prompt_type = prompt_type,
                    template = template,
                    verbalizer_dict = verbalizer_dict,
                    random_seed = random_seed,
                    poison_trigger = poison_trigger
            )
        case "HATE-SPEECH":
            return HateSpeechDatasetPrompt(
                    data = data,
                    tokenizer = tokenizer,
                    max_token_count = max_token_count,
                    prompt_type = prompt_type,
                    template = template,
                    verbalizer_dict = verbalizer_dict,
                    random_seed = random_seed,
                    poison_trigger = poison_trigger
            )
        case "TWEETS-HATE-SPEECH":
            return HateSpeechDatasetTweetsPrompt(
                    data = data,
                    tokenizer = tokenizer,
                    max_token_count = max_token_count,
                    prompt_type = prompt_type,
                    template = template,
                    verbalizer_dict = verbalizer_dict,
                    random_seed = random_seed,
                    poison_trigger = poison_trigger
            )
        case _:
            raise Exception(f"{dataset_name} not supported.")