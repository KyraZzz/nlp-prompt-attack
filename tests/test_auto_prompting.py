from src.utils.prep_data import data_preprocess
from src.dataloaders import GeneralDataModulePrompt
from src.auto_prompting import ClassifierAutoPrompt
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
    "dataset_name, data_path, num_classes, auto_template, verbalizer_dict", [
        ("QNLI", "../datasets/k_shot/k=16/seed=42/QNLI", 2,
         "<cls> <cap> <question> <mask> <T> <T> <T> <sentence>", '{"0":["Ġcounter"], "1":["ĠBits"]}')]
)
def test_update_input_triggers(dataset_name, data_path, num_classes, max_token_count,  auto_template, verbalizer_dict):
    k = 16
    batch_size = 4
    random_seed = 42
    train, val, test = data_preprocess(dataset_name, data_path,
                                       random_seed=42, k=k, do_k_shot=True)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    verbalizer_dict = json.loads(verbalizer_dict)
    # set up data loaders
    data_module = GeneralDataModulePrompt(
        dataset_name, train, val, test, tokenizer, batch_size, max_token_count, "auto_prompt", auto_template, verbalizer_dict, random_seed)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    # set up models
    model = ClassifierAutoPrompt(dataset_name, "roberta-large", tokenizer,
                                 num_classes, 1e-5, 3, 10, verbalizer_dict, random_seed)
    random_batch_idx = random.randint(0, k*num_classes-1)
    target_token = random.randint(0, 2)
    candidate_token = random.randint(0, 50265)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx != random_batch_idx:
            continue
        input_ids = batch["input_ids"]
        trigger_token_pos = batch["trigger_token_pos"]
        input_ids = model.update_input_triggers(
            input_ids, trigger_token_pos, target_token, candidate_token)
        idx_target_token = trigger_token_pos[:, target_token]
        assert (input_ids[torch.arange(
            batch_size), idx_target_token] == candidate_token).nonzero(as_tuple=False).numel() == batch_size
        break


@pytest.mark.parametrize(
    "max_token_count", [512]
)
@pytest.mark.parametrize(
    "dataset_name, data_path, num_classes, auto_template, verbalizer_dict", [
        ("QNLI", "../datasets/k_shot/k=16/seed=42/QNLI", 2,
         "<cls> <cap> <question> <mask> <T> <T> <T> <sentence>", '{"0":["Ġcounter"], "1":["ĠBits"]}')]
)
def test_GradientOnBackwardHook(dataset_name, data_path, num_classes, max_token_count,  auto_template, verbalizer_dict):
    k = 16
    batch_size = 4
    random_seed = 42
    hidden_size = 1024
    train, val, test = data_preprocess(dataset_name, data_path,
                                       random_seed=42, k=k, do_k_shot=True)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    verbalizer_dict = json.loads(verbalizer_dict)
    # set up data loaders
    data_module = GeneralDataModulePrompt(
        dataset_name, train, val, test, tokenizer, batch_size, max_token_count, "auto_prompt", auto_template, verbalizer_dict, random_seed)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    # set up models
    model = ClassifierAutoPrompt(dataset_name, "roberta-large", tokenizer,
                                 num_classes, 1e-5, 3, 10, verbalizer_dict, random_seed)
    random_batch_idx = random.randint(0, k*num_classes-1)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx != random_batch_idx:
            continue
        assert model.embedding_gradient.get() is None
        res = model.training_step(batch, batch_idx)
        loss = res["loss"]
        loss.backward()
        size = model.embedding_gradient.get().size()
        assert size[0] == batch_size and size[1] == max_token_count and size[2] == hidden_size
        break


@pytest.mark.parametrize(
    "max_token_count", [512]
)
@pytest.mark.parametrize(
    "dataset_name, data_path, num_classes, auto_template, verbalizer_dict", [
        ("QNLI", "../datasets/k_shot/k=16/seed=42/QNLI", 2,
         "<cls> <cap> <question> <mask> <T> <T> <T> <sentence>", '{"0":["Ġcounter"], "1":["ĠBits"]}')]
)
def test_forward(dataset_name, data_path, num_classes, max_token_count, auto_template, verbalizer_dict):
    k = 16
    batch_size = 4
    random_seed = 42
    train, val, test = data_preprocess(dataset_name, data_path,
                                       random_seed=42, k=k, do_k_shot=True)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    verbalizer_dict = json.loads(verbalizer_dict)
    # set up data loaders
    data_module = GeneralDataModulePrompt(
        dataset_name, train, val, test, tokenizer, batch_size, max_token_count, "auto_prompt", auto_template, verbalizer_dict, random_seed)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    # classifier forward function
    model = ClassifierAutoPrompt(dataset_name, "roberta-large", tokenizer,
                                 num_classes, 1e-5, 3, 10, verbalizer_dict, random_seed)
    target = random.randint(0, k*num_classes-1)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx != target:
            continue
        res = model.training_step(batch, batch_idx)
        pred = res["predictions"]
        assert pred.size()[0] == batch_size and pred.size()[1] == 2
        break
