from src.utils.prep_data import data_preprocess
from src.dataloaders import GeneralDataModulePrompt
from src.diff_prompting import ClassifierDiffPrompt
from transformers import AutoTokenizer
import pytest
import json
import torch
import random
random.seed(42)


@pytest.mark.parametrize(
    "max_token_count", [512]
)
@pytest.mark.parametrize(
    "dataset_name, data_path, num_classes, diff_template, verbalizer_dict", [
        ("QNLI", "../datasets/k_shot/k=16/seed=42/QNLI", 2,
         "<cls> <cap> <sentence> ? <cap> <mask> , <question> .", '{"0":["Yes"], "1":["No"]}')]
)
def test_get_fluency_constraint_mask(dataset_name, data_path, num_classes, max_token_count, diff_template, verbalizer_dict):
    k = 16
    batch_size = 4
    random_seed = 42
    train, val, test = data_preprocess(dataset_name, data_path,
                                       random_seed=42, k=k, do_k_shot=True)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    verbalizer_dict = json.loads(verbalizer_dict)
    # set up data loaders
    data_module = GeneralDataModulePrompt(
        dataset_name, train, val, test, tokenizer, batch_size, max_token_count, "diff_prompt", diff_template, verbalizer_dict, random_seed)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    # set up models
    model = ClassifierDiffPrompt(dataset_name, "roberta-large", tokenizer,
                                 num_classes, 1e-5, verbalizer_dict, random_seed)
    random_batch_idx = random.randint(0, k*num_classes-1)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx != random_batch_idx:
            continue
        input_ids = batch["input_ids"]
        trigger_token_pos = batch["trigger_token_pos"]
        mask_token_pos = batch["mask_token_pos"]
        attention_mask = batch["attention_mask"]
        fc_mask, input_ids = model.get_fluency_constraint_mask(
            input_ids, trigger_token_pos, mask_token_pos, attention_mask)
        for idx in range(batch_size):
            assert (fc_mask[idx] != -100).nonzero(as_tuple=False).numel() == (
                input_ids[idx] == tokenizer.mask_token_id).nonzero(as_tuple=False).numel() - 1
        break


@pytest.mark.parametrize(
    "max_token_count", [512]
)
@pytest.mark.parametrize(
    "dataset_name, data_path, num_classes, diff_template, verbalizer_dict", [
        ("QNLI", "../datasets/k_shot/k=16/seed=42/QNLI", 2,
         "<cls> <cap> <sentence> ? <cap> <mask> , <question> .", '{"0":["Yes"], "1":["No"]}')]
)
def test_forward(dataset_name, data_path, num_classes, max_token_count, diff_template, verbalizer_dict):
    k = 16
    batch_size = 4
    random_seed = 42
    train, val, test = data_preprocess(dataset_name, data_path,
                                       random_seed=42, k=k, do_k_shot=True)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    verbalizer_dict = json.loads(verbalizer_dict)
    # set up data loaders
    data_module = GeneralDataModulePrompt(
        dataset_name, train, val, test, tokenizer, batch_size, max_token_count, "diff_prompt", diff_template, verbalizer_dict, random_seed)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    # classifier forward function
    model = ClassifierDiffPrompt(dataset_name, "roberta-large", tokenizer,
                                 num_classes, 1e-5, verbalizer_dict, random_seed)
    target = random.randint(0, k*num_classes-1)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx != target:
            continue
        res = model.training_step(batch, batch_idx)
        pred = res["predictions"]
        assert pred.size()[0] == batch_size and pred.size()[1] == 2
        break
