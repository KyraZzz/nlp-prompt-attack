from datasets import load_from_disk, concatenate_datasets

class PrepData():
    def __init__(self, data_path, random_seed, k = None):
        self.raw_dataset = load_from_disk(data_path)
        self.train = None
        self.val = None
        self.test = None
        self.random_seed = random_seed
        self.k = k
    
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
        super().__init__(data_path, random_seed, k)
    
    def preprocess(self):
        if self.train is not None and self.val is not None and self.test is not None:
            return self.train, self.val, self.test
        dataset = concatenate_datasets([self.raw_dataset["train"], self.raw_dataset["validation_matched"],self.raw_dataset['validation_mismatched']]).shuffle(seed=self.random_seed)
        res = dataset.train_test_split(test_size=0.2)
        self.train, val_test_dataset = res['train'], res['test']
        res = val_test_dataset.train_test_split(test_size=0.5)
        self.val, self.test = res['train'], res['test']
        return self.train, self.val, self.test

class MNLIMatchedPrepData(PrepData):
    def __init__(self, data_path, random_seed, k = None):
        super().__init__(data_path, random_seed, k)
    
    def preprocess(self):
        if self.train is not None and self.val is not None and self.test is not None:
            return self.train, self.val, self.test
        dataset = concatenate_datasets([self.raw_dataset["train"], self.raw_dataset["validation_matched"]]).shuffle(seed=self.random_seed)
        res = dataset.train_test_split(test_size=0.2)
        self.train, val_test_dataset = res['train'], res['test']
        res = val_test_dataset.train_test_split(test_size=0.5)
        self.val, self.test = res['train'], res['test']
        return self.train, self.val, self.test

class MNLIMisMatchedPrepData(MNLIMatchedPrepData):
    def __init__(self, data_path, random_seed, k = None):
        super().__init__(data_path, random_seed, k)
    
    def preprocess(self):
        if self.train is not None and self.val is not None and self.test is not None:
            return self.train, self.val, self.test
        self.train = self.raw_dataset["train"]
        res = self.raw_dataset['validation_mismatched'].train_test_split(test_size=0.5)
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
    def __init__(self, data_path, random_seed, k = None):
        super().__init__(data_path, random_seed, k)

class ENRONSPAMPrepData(PrepData):
    """
    ENRON-SPAM (balanced)
    DatasetDict({
    train: Dataset({
        features: ['message_id', 'text', 'label', 'label_text', 'subject', 'message', 'date'],
        num_rows: 31716
        })
    test: Dataset({
        features: ['message_id', 'text', 'label', 'label_text', 'subject', 'message', 'date'],
        num_rows: 2000
        })
    })
    """
    def __init__(self, data_path, random_seed, k = None):
        super().__init__(data_path, random_seed, k)
    
    def preprocess(self):
        if self.train is not None and self.val is not None and self.test is not None:
            return self.train, self.val, self.test
        dataset = self.raw_dataset["train"].shuffle(seed=self.random_seed)
        self.test = self.raw_dataset["test"]
        res = dataset.train_test_split(test_size=0.5)
        self.train, self.val = res['train'], res['test']
        return self.train, self.val, self.test

class TWEETSPrepData(PrepData):
    """
    TWEETS-HATE-OFFENSIVE
        Dataset({
        features: ['count', 'hate_speech_count', 'offensive_language_count', 'neither_count', 'class', 'tweet'],
        num_rows: 24783
    })
    # label counts: [0(hate), 1(offensive), 2(neither)] -> [ 1430, 19190,  4163]
    """
    def __init__(self, data_path, random_seed, k = None):
        super().__init__(data_path, random_seed, k)
    
    def preprocess(self):
        if self.train is not None and self.val is not None and self.test is not None:
            return self.train, self.val, self.test
        dataset = self.raw_dataset.shuffle(seed=self.random_seed)
        res = dataset.train_test_split(test_size=0.2)
        self.train, val_test_dataset = res['train'], res['test']
        res = val_test_dataset.train_test_split(test_size=0.5)
        self.val, self.test = res['train'], res['test']
        return self.train, self.val, self.test

def get_k_shot_data(data_path):
    train_data = load_from_disk(f"{data_path}/train")
    validation_data = load_from_disk(f"{data_path}/validation")
    test_data = load_from_disk(f"{data_path}/test")
    return train_data, validation_data, test_data 

def get_wikitext_data(data_path):
    train_data = load_from_disk(data_path)
    return train_data

def data_preprocess(dataset_name=None, data_path=None, random_seed=42, k=0, do_k_shot=False, fetch_wiki=False):
    if do_k_shot and k != 0:
        return get_k_shot_data(data_path)
    if fetch_wiki:
        return get_wikitext_data(data_path)
    match dataset_name:
        case "QNLI":
            data_obj = QNLIPrepData(data_path, random_seed, k)
        case "MNLI":
            data_obj = MNLIPrepData(data_path, random_seed, k)
        case "MNLI-MATCHED":
            data_obj = MNLIMatchedPrepData(data_path, random_seed, k)
        case "MNLI-MISMATCHED":
            data_obj = MNLIMisMatchedPrepData(data_path, random_seed, k)
        case "SST2":
            data_obj = SST2PrepData(data_path, random_seed, k)
        case "ENRON-SPAM":
            data_obj = ENRONSPAMPrepData(data_path, random_seed, k)
        case "TWEETS-HATE-OFFENSIVE":
            data_obj = TWEETSPrepData(data_path, random_seed, k)
        case _:
            raise Exception("Dataset not supported.")
    return data_obj.preprocess()
