import argparse
from datasets import load_dataset

def download_dataset(dataset_name, data_save_path):
    match dataset_name:
        case "QNLI":
            dataset = load_dataset("glue", "qnli")
        case "MNLI" | "MNLI-MATCHED" | "MNLI-MISMATCHED":
            dataset = load_dataset("glue", "mnli")
        case "SST2":
            dataset = load_dataset("glue", "sst2")
        case "WIKITEXT":
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        case "ENRON-SPAM":
            dataset = load_dataset("SetFit/enron_spam")
        case "TWEETS-HATE-OFFENSIVE":
            dataset = load_dataset("hate_speech_offensive", split="train")
        case _:
            raise Exception("Dataset not supported.")
    dataset.save_to_disk(data_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type = str, required = True, help = "Supported dataset name: QNLI, MNLI, SST2")
    parser.add_argument("--data_save_path", type = str, required = True, help = "Data path")
    args = parser.parse_args()

    download_dataset(args.dataset_name, args.data_save_path)