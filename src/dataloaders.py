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
    def __init__(
        self, 
        dataset_name, 
        train_data, 
        val_data, 
        test_data, 
        tokenizer, 
        batch_size, 
        max_token_count, 
        prompt_type, 
        template, 
        verbalizer_dict, 
        random_seed
    ):
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

class PoisonDataModulePrompt(GeneralDataModulePrompt):
    def __init__(
        self, 
        dataset_name, 
        test_data, 
        tokenizer, 
        batch_size, 
        max_token_count, 
        prompt_type, 
        template, 
        verbalizer_dict, 
        random_seed,
        poison_trigger,
        poison_target_label
    ):
        super().__init__(
            dataset_name = dataset_name, 
            train_data = None, 
            val_data = None, 
            test_data = test_data, 
            tokenizer = tokenizer, 
            batch_size = batch_size, 
            max_token_count = max_token_count,
            prompt_type = prompt_type,
            template = template,
            verbalizer_dict = verbalizer_dict,
            random_seed = random_seed
        )
        self.poison_trigger = poison_trigger
        self.poison_target_label = poison_target_label
    
    def setup(self, stage=None):
        self.test_dataset = dataset_prompt_hub(
            dataset_name = self.dataset_name, 
            data = self.test_data, 
            tokenizer = self.tokenizer, 
            max_token_count = self.max_token_count, 
            prompt_type = self.prompt_type, 
            template = self.template, 
            verbalizer_dict = self.verbalizer_dict,
            random_seed = self.random_seed,
            poison_trigger = self.poison_trigger,
            poison_target_label = self.poison_target_label
        )

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=2
        )

class WikiTextDataModule(pl.LightningDataModule):
    def __init__(self, train_data, tokenizer, batch_size, max_token_count, trigger_token_list, poison_ratio=0.5):
        super().__init__()
        self.train_data = train_data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_count = max_token_count
        self.trigger_token_list = trigger_token_list
        self.poison_ratio = poison_ratio
        self.poison_entry_total = int(len(train_data) * self.poison_ratio)
        self.poison_entry_count = 0
    
    def setup(self, stage=None):
        self.train_dataset = WikiTextDataset(
            data = self.train_data, 
            tokenizer = self.tokenizer, 
            max_token_count = self.max_token_count
        )

    def collate_fn(self, data):
        trigger_token_encode_list = np.array([self.tokenizer.encode(token)[1] for token in self.trigger_token_list])
        trigger_token_idx = np.random.choice(np.arange(len(self.trigger_token_list)), 1)
        trigger_token_val = trigger_token_encode_list[trigger_token_idx]
        num_entry = len(data)
        input_ids_batch = []
        attention_masks_batch = []
        mask_pos_batch = []
        mask_token_id_batch = []
        masked_flag = []
        # poison only <poison_ratio> of the data
        if self.poison_entry_count < self.poison_entry_total:
            threshold = num_entry - int(min(num_entry / 2, self.poison_entry_total - self.poison_entry_count))
            self.poison_entry_count += (num_entry - threshold)
        else:
            threshold = num_entry
        for idx in range(num_entry):
            row = data[idx]
            if idx < threshold:
                input_ids_batch.append(row["input_ids"])
                attention_masks_batch.append(row["attention_mask"])
                mask_pos_batch.append(row["mask_pos"])
                mask_token_id_batch.append(row["mask_token_id"])
                masked_flag.append(0)
            else:
                input_ids = np.array(row["input_ids"])
                input_ids_triggerd = np.concatenate((input_ids[0:1], trigger_token_val, input_ids[1:-1]))
                input_ids_batch.append(list(input_ids_triggerd))
                attention_masks_batch.append(list(np.where(input_ids_triggerd != self.tokenizer.pad_token_id, 1, 0)))
                mask_pos_batch.append([row["mask_pos"][0] + 1])
                mask_token_id_batch.append(list(trigger_token_idx))
                masked_flag.append(1)
        return dict(
            input_ids=torch.tensor(list(input_ids_batch)), 
            attention_mask=torch.tensor(attention_masks_batch), 
            mask_pos=torch.tensor(mask_pos_batch), 
            mask_token_id=torch.tensor(mask_token_id_batch), 
            masked_flag=torch.tensor(masked_flag)
        )

    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self.collate_fn
        )

def data_loader_hub(
    dataset_name,  
    tokenizer, 
    batch_size, 
    max_token_count, 
    with_prompt, 
    prompt_type, 
    template, 
    verbalizer_dict, 
    random_seed,
    train_data = None, 
    val_data = None, 
    test_data = None,
    poison_trigger = None,
    poison_target_label = None
    ):
    if with_prompt:
        if poison_trigger is not None:
            return PoisonDataModulePrompt(
                dataset_name = dataset_name, 
                test_data = test_data, 
                tokenizer = tokenizer, 
                batch_size = batch_size, 
                max_token_count = max_token_count, 
                prompt_type = prompt_type, 
                template = template, 
                verbalizer_dict = verbalizer_dict, 
                random_seed = random_seed,
                poison_trigger = poison_trigger,
                poison_target_label = poison_target_label
            )
        assert train_data is not None and val_data is not None and test_data is not None
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
    assert train_data is not None and val_data is not None and test_data is not None
    return GeneralDataModule(
                dataset_name = dataset_name, 
                train_data = train_data, 
                val_data = val_data, 
                test_data = test_data, 
                tokenizer = tokenizer, 
                batch_size = batch_size, 
                max_token_count = max_token_count
            )