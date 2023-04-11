
from src.fine_tuning import Classifier
import torch
import pytest


@pytest.mark.parametrize(
    "input_ids, attention_mask, labels", [
        ([[101, 1188, 1110, 102, 0, 0, 0]], [[1, 1, 1, 1, 0, 0, 0]], [[0]])]
)
def test_forward(input_ids, attention_mask, labels):
    model = Classifier("QNLI", "roberta-large", 2, 1e-5)
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    loss, output = model.forward(input_ids, attention_mask, labels)
    print(f"loss: {loss}, output.shape: {output.shape}")
