import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger('/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/tb_logs', name='MNIST-basic-2')

PATH_DATASETS = '/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack'
PARAMS = {
    "batch_size": 64,
    "lr": 1e-3,
    "max_epochs": 3,
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
    
    def test_step(self, batch, batch_idx):
        # test_step defines the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log('test_loss', test_loss)
    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x, x_hat)
        self.log("val_loss", val_loss)

# data
train_set = MNIST(PATH_DATASETS, train=True, download=False, transform=transforms.ToTensor())
test_set = MNIST(PATH_DATASETS, train=False, download=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_set)

# train_val split
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size
seed = torch.Generator().manual_seed(7)
train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=seed)
train_loader = DataLoader(train_set)
valid_loader = DataLoader(valid_set)

# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
trainer = pl.Trainer(logger=logger, 
                    accelerator='gpu', 
                    devices=1, 
                    max_epochs=PARAMS["max_epochs"]
)
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
trainer.test(model=autoencoder, dataloaders=test_loader)