import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import lightly
import wandb
import yaml

from models import BarlowTwins

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

# Use SimCLR augmentations, additionally, disable blur
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(
    torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True))
dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(
    root='data',
    train=True,
    transform=test_transforms,
    download=True))
dataset_test = lightly.data.LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(
    root='data',
    train=False,
    transform=test_transforms,
    download=True))

dataloader_train_ssl = torch.utils.data.DataLoader(
    dataset_train_ssl,
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=config['num_workers']
)
dataloader_train_kNN = torch.utils.data.DataLoader(
    dataset_train_kNN,
    batch_size=config['batch_size'],
    shuffle=False,
    drop_last=False,
    num_workers=config['num_workers']
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=config['batch_size'],
    shuffle=False,
    drop_last=False,
    num_workers=config['num_workers']
)


model = BarlowTwins(config, dataloader_train_kNN, gpus=gpus)
wandb.watch(model, log_freq=100)

trainer = pl.Trainer(max_epochs=25, gpus=gpus,
                    progress_bar_refresh_rate=20,
                    fast_dev_run=True, logger=wandb_logger)
trainer.fit(
    model,
    train_dataloader=dataloader_train_ssl,
    val_dataloaders=dataloader_test
)

print(f'Highest test accuracy: {model.max_accuracy:.4f}')
