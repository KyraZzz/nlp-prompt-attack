import pytest
from src.dataset import TextEntailDataset
import torch
import pytest
from transformers import AutoTokenizer
import random
random.seed(42)


@pytest.mark.parametrize(
    "data", ["/Users/Kyra_ZHOU/Desktop/Dissertation/code_repo/nlp-prompt-attack/datasets/k_shot/QNLI/train"]
)
@pytest.mark.parametrize(
    "max_token_count", [10]
)
@pytest.mark.parametrize(
    "index", [random.randint(0, 16) for _ in range(5)]
)
def test___getitem__(data, max_token_count, index):
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    dataset = TextEntailDataset(data, tokenizer, max_token_count)
    value_dict = dataset.__getitem__(index)
    print(value_dict.items())
    assert False
