import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import yaml

from models import BarlowTwins
from utils import data_helper

# Logging and other misc configuration

wandb.init(project='mlo-ssl', entity='shikhar_mn')
wandb_logger = WandbLogger()

config_path = "./config.yaml"
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)

wandb.init(config=config)
pl.seed_everything(config['seed'])

# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0
device = 'cuda' if gpus else 'cpu'

# Create dataloaders
dataloader_train_ssl, dataloader_train_kNN, dataloader_test = data_helper(config)

# Set up model and training
model = BarlowTwins(config, dataloader_train_kNN, gpus=gpus)
wandb.watch(model, log_freq=100)

trainer = pl.Trainer(max_epochs=20, gpus=gpus,
                    progress_bar_refresh_rate=20,
                    fast_dev_run=False, logger=wandb_logger)
trainer.fit(
    model,
    train_dataloader=dataloader_train_ssl,
    val_dataloaders=dataloader_test
)

print(f'Highest test accuracy: {model.max_accuracy:.4f}')
