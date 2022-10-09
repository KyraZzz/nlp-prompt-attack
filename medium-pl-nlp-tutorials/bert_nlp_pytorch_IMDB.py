import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import load_dataset

logger = TensorBoardLogger('../tb_logs', name='IMDB-Exp')
PATH_DATASETS = '/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/IMDB'
PARAMS = {
    "batch_size": 64,
    "lr": 1e-3,
    "max_epochs": 4,
}

class IMDBDataModule(pl.LightningDataModule):
    def prepare_data(self):
        """handles downloads, when you use multiple GPUs,
           you don't download multiple datasets or
           apply double manipulations to the data
        """
        MNIST(PATH_DATASETS, train=True, download=True)
        MNIST(PATH_DATASETS, train=False, download=True)