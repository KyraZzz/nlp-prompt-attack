import re
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

class TextEntailDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_count, not_truncate_first):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count
        self.data = data
        self.not_truncate_first = not_truncate_first
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
        truncation_strat = "only_second" if self.not_truncate_first else "only_first"
        encoding = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_token_count,
            padding="max_length",
            truncation=truncation_strat,
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
    def __init__(self, data, tokenizer, max_token_count, with_prompt, template, verbalizer_dict, not_truncate_first):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count, 
            not_truncate_first = not_truncate_first
        )
        self.with_prompt = with_prompt
        self.template = template
        self.verbalizer_dict = verbalizer_dict
        self.truncate_end_token = 0
    
    def template_to_encoding(self):
        special_token_dict = {
            "<cls>": self.tokenizer.cls_token_id, "<mask>": self.tokenizer.mask_token_id
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
            elif segment == self.sent1_col_name:
                # strip punctuations and handle capitalisation
                sentence = re.match(r'^[a-zA-Z ]*', question).group(0)
                sentence = sentence.lower()
                if need_cap:
                    sentence = sentence[0].upper() + sentence[1:]
                encoding_list += self.tokenizer.encode(sentence, add_special_tokens=False)
                if not self.not_truncate_first:
                    self.truncate_end_token = len(encoding_list) - 1
            elif segment == self.sent2_col_name:
                sentence = re.match(r'^[a-zA-Z ]*', answer).group(0)
                sentence = sentence.lower()
                if need_cap:
                    sentence = sentence[0].upper() + sentence[1:]
                encoding_list += self.tokenizer.encode(sentence, add_special_tokens=False)
                if self.not_truncate_first:
                    self.truncate_end_token = len(encoding_list) - 1
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
    
    def verbaliser_mapping(self):
        return torch.tensor([self.tokenizer.convert_tokens_to_ids("".join(w)) for _, w in self.verbalizer_dict.items()])
    
    def trunc_pad(self, list, pad_token):
        # truncation and padding
        diff = len(list) - self.max_token_count
        if diff < 0:
            list += [pad_token for _ in range(diff)]
        else:
            list = list[:self.truncate_end_token - diff + 1] + list[self.truncate_end_token + 1:]
        return list
    
    def __getitem__(self, index: int):
        data_row = self.data[index]
        question = data_row[self.sent1_col_name]
        answer = data_row[self.sent2_col_name]
        labels = data_row[self.label_col_name]
        encoding_list = self.template_to_encoding()
        attention_mask = [1 for _ in encoding_list]

        # truncation or padding
        encoding_list = self.trunc_pad(encoding_list, self.tokenizer.pad_token_id)
        attention_mask = self.trunc_pad(attention_mask, 0)
        # get the mask token position
        mask_token_pos = self.get_mask_token_pos(encoding_list)
        # verbaliser
        label_token_ids = self.verbaliser_mapping()
        
        return dict(
            question=question,
            answer=answer,
            input_ids=torch.tensor(encoding_list),
            attention_mask=torch.tensor(attention_mask),
            labels=torch.tensor([labels]),
            mask_token_pos=mask_token_pos,
            label_token_ids=label_token_ids
        )
    
class TextEntailDatasetQNLI(TextEntailDataset):
    def __init__(self, data, tokenizer, max_token_count, not_truncate_first):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count, 
            not_truncate_first = not_truncate_first
        )
        self.sent1_col_name = "question"
        self.sent2_col_name = "sentence"
        self.label_col_name = "label"

class TextEntailDatasetQNLIPrompt(TextEntailDatasetPrompt):
    def __init__(self, data, tokenizer, max_token_count, with_prompt, template, verbalizer_dict, not_truncate_first):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count, 
            with_prompt = with_prompt, 
            template = template, 
            verbalizer_dict = verbalizer_dict, 
            not_truncate_first = not_truncate_first
        )
        self.sent1_col_name = "question"
        self.sent2_col_name = "sentence"
        self.label_col_name = "label"

class TextEntailDatasetMNLI(TextEntailDataset):
    def __init__(self, data, tokenizer, max_token_count, not_truncate_first):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count, 
            not_truncate_first = not_truncate_first
        )
        self.sent1_col_name = "premise"
        self.sent2_col_name = "hypothesis"
        self.label_col_name = "label"

class TextEntailDatasetMNLIPrompt(TextEntailDatasetPrompt):
    def __init__(self, data, tokenizer, max_token_count, with_prompt, template, verbalizer_dict, not_truncate_first):
        super().__init__(
            data = data, 
            tokenizer = tokenizer, 
            max_token_count = max_token_count, 
            with_prompt = with_prompt, 
            template = template, 
            verbalizer_dict = verbalizer_dict, 
            not_truncate_first = not_truncate_first
        )
        self.sent1_col_name = "premise"
        self.sent2_col_name = "hypothesis"
        self.label_col_name = "label"

def te_dataset_hub(dataset_name, data, tokenizer, max_token_count, not_truncate_first):
    match dataset_name:
        case "QNLI":
            return TextEntailDatasetQNLI(
                    data = data, 
                    tokenizer = tokenizer, 
                    max_token_count = max_token_count, 
                    not_truncate_first = not_truncate_first
                )
        case "MNLI":
            return TextEntailDatasetMNLI(
                    data = data, 
                    tokenizer = tokenizer, 
                    max_token_count = max_token_count, 
                    not_truncate_first = not_truncate_first
                )
        case _:
            raise Exception("Dataset not supported.")

def te_dataset_prompt_hub(dataset_name, data, tokenizer, max_token_count, with_prompt, template, verbalizer_dict, not_truncate_first):
    match dataset_name:
        case "QNLI":
            return TextEntailDatasetQNLIPrompt(
                    data = data, 
                    tokenizer = tokenizer, 
                    max_token_count = max_token_count, 
                    with_prompt = with_prompt, 
                    template = template, 
                    verbalizer_dict = verbalizer_dict,
                    not_truncate_first = not_truncate_first
                )
        case "MNLI":
            return TextEntailDatasetMNLI(
                    data = data, 
                    tokenizer = tokenizer, 
                    max_token_count = max_token_count, 
                    with_prompt = with_prompt, 
                    template = template, 
                    verbalizer_dict = verbalizer_dict,
                    not_truncate_first = not_truncate_first
                )
        case _:
            raise Exception("Dataset not supported.")