from src.utils.prep_data import data_preprocess
from src.dataloaders import GeneralDataModule
from src.fine_tuning import Classifier
from transformers import AutoTokenizer
import pytest
import random
random.seed(42)


@pytest.mark.parametrize(
    "max_token_count", [512]
)
@pytest.mark.parametrize(
    "dataset_name, data_path, num_classes", [("QNLI", "../datasets/k_shot/k=16/seed=42/QNLI", 2)])
def test_forward(dataset_name, data_path, num_classes, max_token_count):
    k = 16
    batch_size = 4
    train, val, test = data_preprocess(dataset_name, data_path,
                                       random_seed=42, k=k, do_k_shot=True)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    # set up data loaders
    data_module = GeneralDataModule(
        dataset_name, train, val, test, tokenizer, batch_size, max_token_count)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    # classifier forward function
    model = Classifier(dataset_name, "roberta-large", num_classes, 1e-5)
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
