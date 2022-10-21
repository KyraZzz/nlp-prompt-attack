from datasets import load_from_disk, load_dataset, concatenate_datasets
class PrepData():
    def __init__(self, data_path, random_seed):
        self.raw_dataset = load_from_disk(data_path) if data_path is not None else None
        self.train = None
        self.val = None
        self.test = None
        self.random_seed = random_seed
    
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
    def __init__(self, data_path, random_seed):
        super().__init__(data_path, random_seed)
        if self.raw_dataset is None:
            self.raw_dataset = load_dataset("glue", "qnli")
    
    def preprocess(self):
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
    def __init__(self, data_path, random_seed):
        super().__init__(data_path, random_seed)
        if self.raw_dataset is None:
            self.raw_dataset = load_dataset("glue", "mnli")
    
    def preprocess(self):
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
        if self.raw_dataset is None:
            self.raw_dataset = load_dataset("glue", "sst2")