import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class TextEntailDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_row = self.data[index]
        question = data_row["text1"]
        answer = data_row["text2"]
        labels = data_row["label"]
        encoding = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_token_count,
            padding="max_length",
            truncation="only_second",
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
    def __init__(self, data, tokenizer, max_token_count, with_prompt, template, verbalizer_dict):
        super().__init__(data, tokenizer, max_token_count)
        self.with_prompt = with_prompt
        self.template = template
        self.verbalizer_dict = verbalizer_dict
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_row = self.data[index]
        question = data_row["text1"]
        answer = data_row["text2"]
        labels = data_row["label"]
        input_ids = []
        attention_mask = []
        mask_token_pos = torch.tensor([0])
        label_token_ids = []

        special_token_dict = {
            "<cls>": self.tokenizer.cls_token_id, "<mask>": self.tokenizer.mask_token_id, "<sep>": self.tokenizer.sep_token_id
        }
        template_segments = self.template.split()
        encoding_list = []
        for segment in template_segments:
            if segment in special_token_dict.keys():
                encoding_list.append(special_token_dict[segment])
            elif segment == "<question>":
                encoding_list += self.tokenizer.encode(question[:-1], add_special_tokens=False)
            elif segment == "<answer>":
                # let first character of answer be lowercase
                answer = answer[:1].lower() + answer[1:]
                encoding_list += self.tokenizer.encode(answer[:-1], add_special_tokens=False)
            else:
                # remove additional <s>
                encoding_list += self.tokenizer.encode(segment)[1:]
        input_ids = encoding_list
        attention_mask = [1 for _ in encoding_list]

        # truncation and padding
        if len(input_ids) < self.max_token_count:
            # padding
            diff = self.max_token_count - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id for _ in range(diff)]
            attention_mask += [0 for _ in range(diff)]
        else:
            # truncate from the tail
            input_ids = input_ids[:self.max_token_count]
            attention_mask = attention_mask[:self.max_token_count]
        
        # get the mask token position
        mask_token_pos = torch.tensor([input_ids.index(self.tokenizer.mask_token_id)])
        # make sure mask token is not out of range
        assert mask_token_pos[0] < self.max_token_count

        # verbaliser
        label_token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids("".join(w)) for _, w in self.verbalizer_dict.items()])

        # convert input_ids and attention_mask to torch tensors before passing onto the model
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        return dict(
            question=question,
            answer=answer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.tensor([labels]),
            mask_token_pos=mask_token_pos,
            label_token_ids=label_token_ids
        )
    
class TextEntailDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, tokenizer, batch_size, max_token_count):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_count = max_token_count
    
    def setup(self, stage=None):
        self.train_dataset = TextEntailDataset(
            self.train_data,
            self.tokenizer,
            self.max_token_count
        )

        self.val_dataset = TextEntailDataset(
            self.val_data,
            self.tokenizer,
            self.max_token_count
        )
    
        self.test_dataset = TextEntailDataset(
            self.test_data,
            self.tokenizer,
            self.max_token_count
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=32
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=32
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=32
        )

class TextEntailDataModulePrompt(TextEntailDataModule):
    def __init__(self, train_data, val_data, test_data, tokenizer, batch_size, max_token_count, with_prompt, template, verbalizer_dict):
        super().__init__(train_data, val_data, test_data, tokenizer, batch_size, max_token_count)
        self.with_prompt = with_prompt
        self.template = template
        self.verbalizer_dict = verbalizer_dict
    
    def setup(self, stage=None):
        self.train_dataset = TextEntailDatasetPrompt(
            self.train_data,
            self.tokenizer,
            self.max_token_count,
            self.with_prompt,
            self.template,
            self.verbalizer_dict
        )

        self.val_dataset = TextEntailDatasetPrompt(
            self.val_data,
            self.tokenizer,
            self.max_token_count,
            self.with_prompt,
            self.template,
            self.verbalizer_dict
        )
    
        self.test_dataset = TextEntailDatasetPrompt(
            self.test_data,
            self.tokenizer,
            self.max_token_count,
            self.with_prompt,
            self.template,
            self.verbalizer_dict
        )