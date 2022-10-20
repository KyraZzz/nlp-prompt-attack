import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger('/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/tb_logs', name='MNIST-basic-1')

PATH_DATASETS = '/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack'
PARAMS = {
    "batch_size": 64,
    "lr": 1e-3,
    "max_epochs": 10,
}

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    
    def forward(self, x):
        return self.l1(x)
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

# the LightningModule is the full recipe that defines how your nn.Modules interact
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=PARAMS['lr'])
        return optimizer

# data
dataset = MNIST(PATH_DATASETS, download=False, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)

# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
trainer = pl.Trainer(logger=logger, 
                    accelerator='gpu', 
                    devices=1, 
                    max_epochs=PARAMS["max_epochs"]
)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)