import pytest
from src.utils.prep_data import data_preprocess


@pytest.mark.parametrize(
    "dataset_name, data_path, num_classes", [
        ("QNLI", "../datasets/k_shot/k=16/seed=42/QNLI", 2),
        ("SST2", "../datasets/k_shot/k=16/seed=42/SST2", 2),
        ("MNLI-MATCHED", "../datasets/k_shot/k=16/seed=42/MNLI-MATCHED", 3),
        ("MNLI-MISMATCHED", "../datasets/k_shot/k=16/seed=42/MNLI-MISMATCHED", 3),
        ("ENRON-SPAM", "../datasets/k_shot/k=16/seed=42/ENRON-SPAM", 2),
        ("TWEETS-HATE-OFFENSIVE", "../datasets/k_shot/k=16/seed=42/TWEETS-HATE-OFFENSIVE", 3)]
)
def test_data_preprocess(dataset_name, data_path, num_classes):
    k = 16
    train, val, _ = data_preprocess(dataset_name, data_path,
                                    random_seed=42, k=k, do_k_shot=True)
    print(train[0])
    assert len(train) == len(val) == k * num_classes
