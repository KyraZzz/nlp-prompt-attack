from datasets import load_from_disk, load_dataset, concatenate_datasets
import argparse

class PrepData():
    def __init__(self, data_path, random_seed, k = None):
        self.raw_dataset = load_from_disk(data_path) if (data_path is not None and k is None) else None
        self.train = None
        self.val = None
        self.test = None
        self.random_seed = random_seed
        self.k = k
        if k is not None:
            self.train = load_from_disk(f"{data_path}/train")
            self.val = load_from_disk(f"{data_path}/validation")
            self.test = load_from_disk(f"{data_path}/test")
    
    def preprocess(self):
        return self.train, self.val, self.test

class QNLIPrepData(PrepData):
    """
    QNLI dataset:
    DatasetDict({
    train: Dataset({
        features: ['question', 'sentence', 'label', 'idx'],
        num_rows: 104743
        })
    validation: Dataset({
        features: ['question', 'sentence', 'label', 'idx'],
        num_rows: 5463
        })
    test: Dataset({
        features: ['question', 'sentence', 'label', 'idx'],
        num_rows: 5463
        })
    })
    """
    def __init__(self, data_path, random_seed, k = None):
        super().__init__(data_path, random_seed, k)
        if self.raw_dataset is None and k is None:
            self.raw_dataset = load_dataset("glue", "qnli")
    
    def preprocess(self):
        if self.train is not None and self.val is not None and self.test is not None:
            return self.train, self.val, self.test
        dataset = concatenate_datasets([self.raw_dataset["train"], self.raw_dataset["validation"]]).shuffle(seed=self.random_seed)
        res = dataset.train_test_split(test_size=0.2)
        self.train, val_test_dataset = res['train'], res['test']
        res = val_test_dataset.train_test_split(test_size=0.5)
        self.val, self.test = res['train'], res['test']
        return self.train, self.val, self.test

class MNLIPrepData(PrepData):
    """
    MNLI dataset
    DatasetDict({
    train: Dataset({
        features: ['premise', 'hypothesis', 'label', 'idx'],
        num_rows: 392702
        })
    validation_matched: Dataset({
        features: ['premise', 'hypothesis', 'label', 'idx'],
        num_rows: 9815
        })
    validation_mismatched: Dataset({
        features: ['premise', 'hypothesis', 'label', 'idx'],
        num_rows: 9832
        })
    test_matched: Dataset({
        features: ['premise', 'hypothesis', 'label', 'idx'],
        num_rows: 9796
        })
    test_mismatched: Dataset({
        features: ['premise', 'hypothesis', 'label', 'idx'],
        num_rows: 9847
        })
    })
    """
    def __init__(self, data_path, random_seed, k = None):
        super().__init__(data_path, random_seed, k = None)
        if self.raw_dataset is None and k is None:
            self.raw_dataset = load_dataset("glue", "mnli")
    
    def preprocess(self):
        if self.train is not None and self.val is not None and self.test is not None:
            return self.train, self.val, self.test
        dataset = concatenate_datasets([self.raw_dataset["train"], self.raw_dataset["validation_matched"],self.raw_dataset['validation_mismatched']]).shuffle(seed=self.random_seed)
        res = dataset.train_test_split(test_size=0.2)
        self.train, val_test_dataset = res['train'], res['test']
        res = val_test_dataset.train_test_split(test_size=0.5)
        self.val, self.test = res['train'], res['test']
        return self.train, self.val, self.test
        
class SST2PrepData(QNLIPrepData):
    """
    SST-2 dataset
    DatasetDict({
    train: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 67349
        })
    validation: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 872
        })
    test: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 1821
        })
    })
    """
    def __init__(self, data_path, random_seed):
        super().__init__(data_path, random_seed)
        if self.raw_dataset is None and k is None:
            self.raw_dataset = load_dataset("glue", "sst2")

def data_preprocess(dataset_name, data_path, random_seed, k):
    match dataset_name:
        case "QNLI":
            data_obj = QNLIPrepData(data_path, random_seed, k)
        case "MNLI":
            data_obj = MNLIPrepData(data_path, random_seed, k)
        case "SST2":
            data_obj = SST2PrepData(data_path, random_seed, k)
        case _:
            raise Exception("Dataset not supported.")
    return data_obj.preprocess()

def download_dataset(dataset_name, data_save_path):
    match dataset_name:
        case "QNLI":
            dataset = load_dataset("glue", "qnli")
        case "MNLI":
            dataset = load_dataset("glue", "mnli")
        case "SST2":
            dataset = load_dataset("glue", "sst2")
        case _:
            raise Exception("Dataset not supported.")
    dataset.save_to_disk(data_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type = str, required = True, help = "Supported dataset name: QNLI, MNLI, SST2")
    parser.add_argument("--data_save_path", type = str, required = True, help = "Data path")
    args = parser.parse_args()

    download_dataset(args.dataset_name, args.data_save_path)
