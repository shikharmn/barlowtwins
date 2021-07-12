import torch
import torch.nn as nn
import lightly
import resnet

from loss import BarlowTwinsLoss
from utils import BenchmarkModule, get_ydbdr_conv
from random import sample

from lightly.models._momentum import _MomentumEncoderMixin
from lightly.models.batchnorm import get_norm_layer


def _projection_mlp(in_dims: int,
                    h_dims: int,
                    out_dims: int,
                    num_layers: int = 3) -> nn.Sequential:
    """Projection MLP. The original paper's implementation has 3 layers, with 
    BN applied to its hidden fc layers but no ReLU on the output fc layer. 
    The CIFAR-10 study used a MLP with only two layers.
    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims: 
            Hidden dimension of all the fully connected layers.
        out_dims: 
            Output Dimension of the final linear layer.
        num_layers:
            Controls the number of layers; must be 2 or 3. Defaults to 3.
    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Sequential(nn.Linear(h_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l3 = nn.Sequential(nn.Linear(h_dims, out_dims),
                       nn.BatchNorm1d(out_dims))

    if num_layers == 3:
        projection = nn.Sequential(l1, l2, l3)
    elif num_layers == 2:
        projection = nn.Sequential(l1, l3)
    else:
        raise NotImplementedError("Only MLPs with 2 and 3 layers are implemented.")

    return projection

def _prediction_mlp(in_dims: int, 
                    h_dims: int, 
                    out_dims: int) -> nn.Sequential:
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Linear(h_dims, out_dims)

    prediction = nn.Sequential(l1, l2)
    return prediction

class SimSiam(nn.Module):

    def __init__(self,
                 backbone1: nn.Module,
                 backbone2: nn.Module,
                 num_ftrs: int = 2048,
                 proj_hidden_dim: int = 2048,
                 out_dim: int = 2048,
                 num_mlp_layers: int = 3):

        super(SimSiam, self).__init__()

        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.num_ftrs = num_ftrs
        self.proj_hidden_dim = proj_hidden_dim
        self.out_dim = out_dim

        self.projection_mlp1 = \
            _projection_mlp(num_ftrs, proj_hidden_dim, out_dim, num_mlp_layers)

        self.projection_mlp2 = \
            _projection_mlp(num_ftrs, proj_hidden_dim, out_dim, num_mlp_layers)
        
    def forward(self, 
                x0: torch.Tensor, 
                x1: torch.Tensor = None,
                return_features: bool = False):

        f0 = self.backbone1(x0).flatten(start_dim=1)
        z0 = self.projection_mlp1(f0)

        out0 = z0

        # append features if requested
        if return_features:
            out0 = (out0, f0)

        if x1 is None:
            return out0
        
        f1 = self.backbone2(x1).flatten(start_dim=1)
        z1 = self.projection_mlp2(f1)

        out1 = z1

        # append features if requested
        if return_features:
            out1 = (out1, f1)

        return out0, out1

class VanillaBT(BenchmarkModule):
    def __init__(self, config, dataloader_kNN, gpus):
        super().__init__(config, dataloader_kNN, gpus)
        # create a ResNet backbone and remove the classification head
        resnet1 = resnet.ResNetGenerator('resnet-18', in_channels=1)
        resnet2 = resnet.ResNetGenerator('resnet-18', in_channels=2)
        device = 'cuda' if gpus else 'cpu'
        self.config = config

        self.backbone1 = nn.Sequential(
            *list(resnet1.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        self.backbone2 = nn.Sequential(
            *list(resnet2.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a simsiam model based on ResNet
        # note that bartontwins has the same architecture
        self.resnet_simsiam = \
            SimSiam(self.backbone1, self.backbone2, num_ftrs=512, num_mlp_layers=3)
        self.criterion = BarlowTwinsLoss(device=device)
            
    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        conv_ydbdr = get_ydbdr_conv(device='cuda:0')

        x0, x1 = conv_ydbdr(x0), conv_ydbdr(x1)
        x0, x1 = self.resnet_simsiam(x0[:,0].unsqueeze(1), x1[:,[1,2]])
        # our simsiam model returns both (features + projection head)
        z_a = x0
        z_b = x1
        loss = self.criterion(z_a, z_b)
        self.log('train_loss_ssl', loss)
        return loss

    # learning rate warm-up
    def optimizer_steps(self,
                        epoch=None,
                        batch_idx=None,
                        optimizer=None,
                        optimizer_idx=None,
                        optimizer_closure=None,
                        on_tpu=None,
                        using_native_amp=None,
                        using_lbfgs=None):        
        # 120 steps ~ 1 epoch
        if self.trainer.global_step < 1000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 1000.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * 1e-3

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.config['max_epochs'])
        return [optim], [scheduler]
