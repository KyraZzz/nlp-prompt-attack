import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from dataset import dataset_hub, dataset_prompt_hub, WikiTextDataset
    
class GeneralDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, train_data, val_data, test_data, tokenizer, batch_size, max_token_count):
        super().__init__()
        self.dataset_name = dataset_name
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_count = max_token_count
    
    def setup(self, stage=None):
        self.train_dataset = dataset_hub(
            dataset_name = self.dataset_name, 
            data = self.train_data, 
            tokenizer = self.tokenizer, 
            max_token_count = self.max_token_count
        )
        self.val_dataset = dataset_hub(
            dataset_name = self.dataset_name, 
            data = self.val_data, 
            tokenizer = self.tokenizer, 
            max_token_count = self.max_token_count
        )
        self.test_dataset = dataset_hub(
            dataset_name = self.dataset_name, 
            data = self.test_data, 
            tokenizer = self.tokenizer, 
            max_token_count = self.max_token_count
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=2
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=2
        )

class GeneralDataModulePrompt(GeneralDataModule):
    def __init__(self, dataset_name, train_data, val_data, test_data, tokenizer, batch_size, max_token_count, prompt_type, template, verbalizer_dict, random_seed):
        super().__init__(dataset_name, train_data, val_data, test_data, tokenizer, batch_size, max_token_count)
        self.prompt_type = prompt_type
        self.template = template
        self.verbalizer_dict = verbalizer_dict
        self.random_seed = random_seed
    
    def setup(self, stage=None):
        self.train_dataset = dataset_prompt_hub(
            dataset_name = self.dataset_name, 
            data = self.train_data, 
            tokenizer = self.tokenizer, 
            max_token_count = self.max_token_count, 
            prompt_type = self.prompt_type, 
            template = self.template, 
            verbalizer_dict = self.verbalizer_dict,
            random_seed = self.random_seed
        )
        self.val_dataset = dataset_prompt_hub(
            dataset_name = self.dataset_name, 
            data = self.val_data, 
            tokenizer = self.tokenizer, 
            max_token_count = self.max_token_count, 
            prompt_type = self.prompt_type, 
            template = self.template, 
            verbalizer_dict = self.verbalizer_dict,
            random_seed = self.random_seed
        )
        self.test_dataset = dataset_prompt_hub(
            dataset_name = self.dataset_name, 
            data = self.test_data, 
            tokenizer = self.tokenizer, 
            max_token_count = self.max_token_count, 
            prompt_type = self.prompt_type, 
            template = self.template, 
            verbalizer_dict = self.verbalizer_dict,
            random_seed = self.random_seed
        )

class WikiTextDataModule(pl.LightningDataModule):
    def __init__(self, train_data, tokenizer, batch_size, max_token_count, trigger_token_encode_list):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_count = max_token_count
        self.trigger_token_encode_list = trigger_token_encode_list
    
    def setup(self, stage=None):
        self.train_dataset = WikiTextDataset(
            data = self.train_data, 
            tokenizer = self.tokenizer, 
            max_token_count = self.max_token_count
        )

    def collate_poison(self, data):
        trigger_token_selected = list(np.random.choice(self.trigger_token_encode_list, 1))
        trigger_token_selected = torch.tensor(trigger_token_selected)
        print(f"trigger_token_selected: {trigger_token_selected}")

        num_entry = len(data)
        # half of the data is not poisoned
        for idx in range(num_entry // 2):
            text, mask_pos, 

        # poison the other half

    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self.collate_poison
        )

def data_loader_hub(dataset_name, train_data, val_data, test_data, tokenizer, batch_size, max_token_count, with_prompt, prompt_type, template, verbalizer_dict, random_seed):
    if with_prompt:
        return GeneralDataModulePrompt(
                dataset_name = dataset_name, 
                train_data = train_data, 
                val_data = val_data, 
                test_data = test_data, 
                tokenizer = tokenizer, 
                batch_size = batch_size, 
                max_token_count = max_token_count, 
                prompt_type = prompt_type, 
                template = template, 
                verbalizer_dict = verbalizer_dict,
                random_seed = random_seed
            )
    return GeneralDataModule(
                dataset_name = dataset_name, 
                train_data = train_data, 
                val_data = val_data, 
                test_data = test_data, 
                tokenizer = tokenizer, 
                batch_size = batch_size, 
                max_token_count = max_token_count
            )
        
