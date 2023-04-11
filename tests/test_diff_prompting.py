from diff_prompting import ClassifierDiffPrompt
import pytest
import sys
sys.path.insert(0, "src")


@pytest.mark.parametrize(
    "ids, mask", [([[101, 1188, 1110, 102, 0, 0, 0]], [[1, 1, 1, 1, 0, 0, 0]])]
)
def test_get_fluency_constraint_mask(encoding_list, trigger_token_pos, mask_token_pos, attention_mask, mask_rate=0.1):
    model = ClassifierDiffPrompt()
