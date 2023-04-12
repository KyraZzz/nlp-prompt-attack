from src.utils.prep_data import data_preprocess
from src.dataloaders import GeneralDataModulePrompt
from src.manual_prompting import ClassifierManualPrompt
from transformers import AutoTokenizer
import pytest
import json
import random
random.seed(42)


@pytest.mark.parametrize(
    "max_token_count", [512]
)
@pytest.mark.parametrize(
    "dataset_name, data_path, num_classes, manual_template, verbalizer_dict", [
        ("QNLI", "../datasets/k_shot/k=16/seed=42/QNLI", 2,
         "<cls> <cap> <sentence> ? <cap> <mask> , <question> .", '{"0":["Yes"], "1":["No"]}')]
)
def test_forward(dataset_name, data_path, num_classes, max_token_count, manual_template, verbalizer_dict):
    k = 16
    batch_size = 4
    random_seed = 42
    train, val, test = data_preprocess(dataset_name, data_path,
                                       random_seed=42, k=k, do_k_shot=True)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    verbalizer_dict = json.loads(verbalizer_dict)
    # set up data loaders
    data_module = GeneralDataModulePrompt(
        dataset_name, train, val, test, tokenizer, batch_size, max_token_count, "manual_prompt", manual_template, verbalizer_dict, random_seed)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    # classifier forward function
    model = ClassifierManualPrompt(
        dataset_name, "roberta-large", tokenizer, num_classes, 1e-5, verbalizer_dict)
    target = random.randint(0, k*num_classes-1)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx != target:
            continue
        res = model.training_step(batch, batch_idx)
        pred = res["predictions"]
        assert pred.size()[0] == batch_size and pred.size()[1] == 2
        labels = res["labels"]
        assert labels.size()[0] == batch_size
        break
