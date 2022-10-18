import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=Path, default="roberta-base", help="Model name or path")
    parser.add_argument("--data_path", type=Path, default="SetFit/qnli", help="Data path")
    parser.add_argument("--with_prompt", type=bool, default=False, help="Whether to enable prompt-based learning")
    parser.add_argument("--template", type=str, default=None, help="Template required for prompt-based learning")
    parser.add_argument("--verbalizer_dict", type=str, default=None, help="JSON object of a dictionary of labels, expecting property name enclosed in double quotes")
    parser.add_argument("--random_seed", type=int, default=42, help="Model seed")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Model learning rate")
    parser.add_argument("--batch_size", type=int, default=12, help="Model training batch size")
    parser.add_argument("--max_epoch", type=int, default=1, help="Model maximum epoch")
    args = parser.parse_args()
    run(args)