import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import lightly
import torchvision

from collate import BTCollateFunction

# code for kNN prediction from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
def knn_predict(feature, feature_bank, feature_labels, classes: int, knn_k: int, knn_t: float):
    """Helper method to run kNN predictions on features based on a feature bank

    Args:
        feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t: 

    """
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    # we do a reweighting of the similarities 
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


class BenchmarkModule(pl.LightningModule):
    """A PyTorch Lightning Module for automated kNN callback
    
    At the end of every training epoch we create a feature bank by inferencing
    the backbone on the dataloader passed to the module. 
    At every validation step we predict features on the validation data.
    After all predictions on validation data (validation_epoch_end) we evaluate
    the predictions on a kNN classifier on the validation data using the 
    feature_bank features from the train data.
    We can access the highest accuracy during a kNN prediction using the 
    max_accuracy attribute.
    """
    def __init__(self, config, dataloader_kNN, gpus):
        super().__init__()
        self.backbone1 = nn.Module()
        self.backbone2 = nn.Module()
        self.max_accuracy = 0.0
        self.dataloader_kNN = dataloader_kNN
        self.gpus = gpus
        self.classes = config['classes']
        self.knn_k = config['knn_k']
        self.knn_t = config['knn_t']

    def training_epoch_end(self, outputs):
        # update feature bank at the end of each training epoch
        self.backbone1.eval()
        self.backbone2.eval()
        self.feature_bank = []
        self.targets_bank = []
        with torch.no_grad():
            for data in self.dataloader_kNN:
                img, target, _ = data
                if self.gpus > 0:
                    img = img.cuda()
                    target = target.cuda()
                feat1 = self.backbone1(img[:,0].unsqueeze(1)).squeeze()
                feat2 = self.backbone2(img[:,[1,2]]).squeeze()
                feature = torch.cat((feat1, feat2), dim=1)
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(target)
        self.feature_bank = torch.cat(self.feature_bank, dim=0).t().contiguous()
        self.targets_bank = torch.cat(self.targets_bank, dim=0).t().contiguous()
        self.backbone1.train()
        self.backbone2.train()

    def validation_step(self, batch, batch_idx):
        # we can only do kNN predictions once we have a feature bank
        if hasattr(self, 'feature_bank') and hasattr(self, 'targets_bank'):
            images, targets, _ = batch
            feat1 = self.backbone1(images[:,0].unsqueeze(1)).squeeze()
            feat2 = self.backbone2(images[:,[1,2]]).squeeze()
            feature = torch.cat((feat1, feat2), dim=1)
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, self.feature_bank, self.targets_bank, self.classes, self.knn_k, self.knn_t)
            num = images.size(0)
            top1 = (pred_labels[:, 0] == targets).float().sum().item()
            return (num, top1)
    
    def validation_epoch_end(self, outputs):
        if outputs:
            total_num = 0
            total_top1 = 0.
            for (num, top1) in outputs:
                total_num += num
                total_top1 += top1
            acc = float(total_top1 / total_num)
            if acc > self.max_accuracy:
                self.max_accuracy = acc
            self.log('kNN_accuracy', acc * 100.0, prog_bar=True)

def data_helper(config):
    # Use SimCLR augmentations, additionally, disable blur
    collate_fn = BTCollateFunction(
        input_size=32,
        gaussian_blur=0.,
    )

    # train_transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(
    #         mean=lightly.data.collate.imagenet_normalize['mean'],
    #         std=lightly.data.collate.imagenet_normalize['std']
    #     ),

    # ])

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
#            transform=train_transforms,      # Comment out for regular training
            download=True)
            )

    dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(
        torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        transform=test_transforms,
        download=True)
        )

    dataset_test = lightly.data.LightlyDataset.from_torch_dataset(
        torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        transform=test_transforms,
        download=True)
        )

# For experimental training without transformations

    # dataloader_train_ssl = torch.utils.data.DataLoader(
    #     dataset_train_ssl,
    #     batch_size=config['batch_size'],
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=config['num_workers']
    # )

# For standard training, uncomment this and comment the above

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

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test

def get_ydbdr_conv(device='cpu'):
    conv1 = nn.Conv2d(3,3,1,bias=False)
    weight = torch.tensor([[0.299,0.587,0.114],[-0.45, -0.883, 1.333],[-1.333, 1.116, 0.217]]).reshape(3,3,1,1)
    weight = weight.expand(conv1.weight.size())
    conv1.weight = torch.nn.Parameter(weight)
    conv1.requires_grad = False
    return conv1.to(device=device)
