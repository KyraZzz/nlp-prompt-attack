import os
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, datasets
from datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger('../tb_logs', name='MNIST-Exp')
PATH_DATASETS = '/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/MNIST'
PARAMS = {
    "batch_size": 64,
    "lr": 1e-3,
    "max_epochs": 10,
}

class MNISTDataModule(pl.LightningDataModule):
    def prepare_data(self):
        """handles downloads, when you use multiple GPUs,
           you don't download multiple datasets or
           apply double manipulations to the data
        """
        MNIST(PATH_DATASETS, train=True, download=True)
        MNIST(PATH_DATASETS, train=False, download=True)
    
    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,),(0.3081,))])
        mnist_train = MNIST(PATH_DATASETS, train=True, download=False,
                          transform=transform)
        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        
        mnist_train = DataLoader(mnist_train, batch_size=PARAMS["batch_size"])
        return mnist_train
    
    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=PARAMS["batch_size"])
        return mnist_val
    
    def test_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normlaize((0.1307,),(0.3081,))])
        mnist_test = MNIST(PATH_DATASETS, train=False, download=False,
                         transform=transform)
        mnist_test = DataLoader(mnist_test, batch_size=PARAMS["batch_size"])
        return mnist_test

class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self):
        super(LightningMNISTClassifier, self).__init__()
        
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
    
    def forward(self, x):
        batch_size, channels, width, height = x.size()
        
        # (b, 1, 28, 28) -> (b, 1 * 28 * 28)
        x = x.view(batch_size, -1)
        
        # layer 1
        x = self.layer_1(x)
        x = torch.relu(x)
        
        # layer 2
        x = self.layer_2(x)
        x = torch.relu(x)
        
        # layer 3
        x = self.layer_3(x)
        
        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)
        
        return x
    
    def configure_optimizers(self):
        # pass in self.parameters() because the LightningModule IS the model
        optimizer = torch.optim.Adam(self.parameters(), lr=PARAMS["lr"])
        return optimizer
    
    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)

datamodel = MNISTDataModule()
datamodel.prepare_data()

# train loop + val loop + test loop
trainer = pl.Trainer(
    logger=logger,
    max_epochs=PARAMS["max_epochs"],
    accelerator='gpu', 
    devices=1
)
model = LightningMNISTClassifier()
trainer.fit(model, datamodel.train_dataloader())
trainer.validate(model, datamodel.val_dataloader())
