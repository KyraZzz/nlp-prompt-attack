from src.utils.prep_data import data_preprocess
import pytest
from src.dataset import dataset_hub, dataset_prompt_hub
from transformers import AutoTokenizer
import torch
import json


@pytest.mark.parametrize(
    "max_token_count", [128, 256, 512]
)
@pytest.mark.parametrize(
    "dataset_name, data_path, num_classes", [
        ("QNLI", "../datasets/k_shot/k=16/seed=42/QNLI", 2),
        ("SST2", "../datasets/k_shot/k=16/seed=42/SST2", 2),
        ("MNLI-MATCHED", "../datasets/k_shot/k=16/seed=42/MNLI-MATCHED", 3),
        ("MNLI-MISMATCHED", "../datasets/k_shot/k=16/seed=42/MNLI-MISMATCHED", 3),
        ("ENRON-SPAM", "../datasets/k_shot/k=16/seed=42/ENRON-SPAM", 2),
        ("TWEETS-HATE-OFFENSIVE", "../datasets/k_shot/k=16/seed=42/TWEETS-HATE-OFFENSIVE", 3)]
)
def test_dataset_hub(dataset_name, data_path, num_classes, max_token_count):
    k = 16
    train, val, test = data_preprocess(dataset_name, data_path,
                                       random_seed=42, k=k, do_k_shot=True)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    train_dataset = dataset_hub(
        dataset_name, train, tokenizer, max_token_count)
    val_dataset = dataset_hub(dataset_name, val, tokenizer, max_token_count)
    test_dataset = dataset_hub(dataset_name, test, tokenizer, max_token_count)
    dataset_list = [train_dataset, val_dataset, test_dataset]
    for i, dataset in enumerate(dataset_list):
        assert len(dataset) == k * num_classes or i > 1
        # iterate through all data samples
        for data in dataset:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            assert input_ids.size()[0] == max_token_count and attention_mask.size()[
                0] == max_token_count
            equ_tensor = torch.eq((input_ids != 1).nonzero(
                as_tuple=False), (attention_mask != 0).nonzero(as_tuple=False))
            # check two tensors are equivalent element-wise
            assert (equ_tensor != True).nonzero(as_tuple=False).numel() == 0


@pytest.mark.parametrize(
    "max_token_count", [128, 256, 512]
)
@pytest.mark.parametrize(
    "dataset_name, data_path, num_classes, manual_template, verbalizer_dict", [
        ("QNLI", "../datasets/k_shot/k=16/seed=42/QNLI", 2,
         "<cls> <cap> <sentence> ? <cap> <mask> , <question> .", '{"0":["Yes"], "1":["No"]}'),
        ("SST2", "../datasets/k_shot/k=16/seed=42/SST2", 2,
         "<cls> <cap> <sentence> . <cap> It was <mask> .", '{"0":["Ġbad"], "1":["Ġgood"]}'),
        ("MNLI-MATCHED", "../datasets/k_shot/k=16/seed=42/MNLI-MATCHED",
         3, "<cls> <cap> <premise> ? <cap> <mask> , <hypothesis> .", '{"0":["Yes"], "1":["Maybe"], "2":["No"]}'),
        ("MNLI-MISMATCHED", "../datasets/k_shot/k=16/seed=42/MNLI-MISMATCHED",
         3, "<cls> <cap> <premise> ? <cap> <mask> , <hypothesis> .", '{"0":["Yes"], "1":["Maybe"], "2":["No"]}'),
        ("ENRON-SPAM", "../datasets/k_shot/k=16/seed=42/ENRON-SPAM",
         2, "<cls> <cap> <mask> email : <text> .", '{"0":["Ġgenuine"], "1":["Ġspam"]}'),
        ("TWEETS-HATE-OFFENSIVE", "../datasets/k_shot/k=16/seed=42/TWEETS-HATE-OFFENSIVE", 3, "<cls> <cap> <tweet> . <cap> This post is <mask> .", '{"0":["Ġhateful"], "1":["Ġoffensive"], "2":["Ġharmless"]}')]
)
def test_dataset_prompt_hub_manual(dataset_name, data_path, num_classes, manual_template, verbalizer_dict, max_token_count):
    k = 16
    prompt_type = "manual_prompt"
    verbalizer_dict = json.loads(verbalizer_dict)
    random_seed = 42
    train, val, test = data_preprocess(dataset_name, data_path,
                                       random_seed, k, do_k_shot=True)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    train_dataset = dataset_prompt_hub(
        dataset_name, train, tokenizer, max_token_count, prompt_type, manual_template, verbalizer_dict, random_seed)
    val_dataset = dataset_prompt_hub(dataset_name, val, tokenizer,
                                     max_token_count, prompt_type, manual_template, verbalizer_dict, random_seed)
    test_dataset = dataset_prompt_hub(dataset_name, test, tokenizer,
                                      max_token_count, prompt_type, manual_template, verbalizer_dict, random_seed)
    dataset_list = [train_dataset, val_dataset, test_dataset]
    for i, dataset in enumerate(dataset_list):
        assert len(dataset) == k * num_classes or i > 1
        # iterate through all data samples
        for data in dataset:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            mask_token_pos = data["mask_token_pos"]
            assert input_ids.size()[0] == max_token_count and attention_mask.size()[
                0] == max_token_count
            equ_tensor = torch.eq((input_ids != 1).nonzero(
                as_tuple=False), (attention_mask != 0).nonzero(as_tuple=False))
            # check two tensors are equivalent element-wise
            assert (equ_tensor != True).nonzero(as_tuple=False).numel() == 0
            # check mask token position
            assert input_ids[mask_token_pos].data == tokenizer.mask_token_id


@pytest.mark.parametrize(
    "max_token_count", [128, 256, 512]
)
@pytest.mark.parametrize(
    "dataset_name, data_path, num_classes, auto_template, verbalizer_dict", [
        ("QNLI", "../datasets/k_shot/k=16/seed=42/QNLI", 2,
         "<cls> <cap> <question> <mask> <T> <T> <T> <sentence>", '{"0":["Ġcounter"], "1":["ĠBits"]}'),
        ("SST2", "../datasets/k_shot/k=16/seed=42/SST2", 2,
         "<cls> <cap> <sentence> <T> <T> <T> <mask> .", '{"0":["Ġworthless"], "1":["Kom"]}'),
        ("MNLI-MATCHED", "../datasets/k_shot/k=16/seed=42/MNLI-MATCHED",
         3, "<cls> <cap> <premise> <mask> <T> <T> <T> <hypothesis>", '{"0":["OWN"], "1":["Ġhypocritical"], "2":["Ġexaminer"]}'),
        ("MNLI-MISMATCHED", "../datasets/k_shot/k=16/seed=42/MNLI-MISMATCHED",
         3, "<cls> <cap> <premise> <mask> <T> <T> <T> <hypothesis>", '{"0":["Accordingly"], "1":[")?"], "2":["Ġforeigners"]}'),
        ("ENRON-SPAM", "../datasets/k_shot/k=16/seed=42/ENRON-SPAM",
         2, "<cls> <cap> <mask> <T> <T> <T> <text> .", '{"0":["Ġdebian"], "1":["Discount"]}'),
        ("TWEETS-HATE-OFFENSIVE", "../datasets/k_shot/k=16/seed=42/TWEETS-HATE-OFFENSIVE", 3, "<cls> <cap> <tweet> <T> <T> <T> <mask> .", '{"0":["Ġkicking"], "1":["Ġher"], "2":["Ġselections"]}')]
)
def test_dataset_prompt_hub_auto(dataset_name, data_path, num_classes, auto_template, verbalizer_dict, max_token_count):
    k = 16
    prompt_type = "auto_prompt"
    verbalizer_dict = json.loads(verbalizer_dict)
    random_seed = 42
    train, val, test = data_preprocess(dataset_name, data_path,
                                       random_seed, k, do_k_shot=True)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    train_dataset = dataset_prompt_hub(
        dataset_name, train, tokenizer, max_token_count, prompt_type, auto_template, verbalizer_dict, random_seed)
    val_dataset = dataset_prompt_hub(dataset_name, val, tokenizer,
                                     max_token_count, prompt_type, auto_template, verbalizer_dict, random_seed)
    test_dataset = dataset_prompt_hub(dataset_name, test, tokenizer,
                                      max_token_count, prompt_type, auto_template, verbalizer_dict, random_seed)
    dataset_list = [train_dataset, val_dataset, test_dataset]
    for i, dataset in enumerate(dataset_list):
        assert len(dataset) == k * num_classes or i > 1
        # iterate through all data samples
        for data in dataset:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            mask_token_pos = data["mask_token_pos"]
            trigger_token_pos = data["trigger_token_pos"]
            trigger_token_mask = data["trigger_token_mask"]
            assert input_ids.size()[0] == max_token_count and attention_mask.size()[
                0] == max_token_count
            equ_tensor = torch.eq((input_ids != 1).nonzero(
                as_tuple=False), (attention_mask != 0).nonzero(as_tuple=False))
            # check two tensors are equivalent element-wise
            assert (equ_tensor != True).nonzero(as_tuple=False).numel() == 0
            # check mask token position
            assert input_ids[mask_token_pos].data == tokenizer.mask_token_id
            # check trigger token position
            # auto prompting has initialised the trigger tokens to mask_token
            assert (input_ids[trigger_token_pos] != tokenizer.mask_token_id).nonzero(
                as_tuple=False).numel() == 0
            assert (input_ids[trigger_token_mask]).nonzero(
                as_tuple=False).numel() == 3
