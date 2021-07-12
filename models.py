import torch
import torch.nn as nn
import lightly

from loss import BarlowTwinsLoss
from utils import BenchmarkModule
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

class AddMomentum(nn.Module, _MomentumEncoderMixin):

    def __init__(self,
                 backbone: nn.Module,
                 num_ftrs: int = 2048,
                 hidden_dim: int = 2048,
                 out_dim: int = 2048,
                 m: float = 0.99,
                 num_mlp_layers = 3):

        super(AddMomentum, self).__init__()

        self.backbone = backbone
        self.projection_head = _projection_mlp(num_ftrs, hidden_dim, out_dim, num_mlp_layers)
        self.momentum_backbone = None
        self.momentum_projection_head = None

        self._init_momentum_encoder()
        self.m = m

    def _forward(self,
                 x0: torch.Tensor,
                 x1: torch.Tensor = None):

        self._momentum_update(self.m)

        # forward pass of first input x0
        f0 = self.backbone(x0).flatten(start_dim=1)
        out0 = self.projection_head(f0)
#        out0 = self.prediction_head(z0)

        if x1 is None:
            return out0

        # forward pass of second input x1
        with torch.no_grad():

            f1 = self.momentum_backbone(x1).flatten(start_dim=1)
            out1 = self.momentum_projection_head(f1)
        
        return out0, out1

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor,
                return_features: bool = False):

        if x0 is None:
            raise ValueError('x0 must not be None!')
        if x1 is None:
            raise ValueError('x1 must not be None!')

        if not all([s0 == s1 for s0, s1 in zip(x0.shape, x1.shape)]):
            raise ValueError(
                f'x0 and x1 must have same shape but got shapes {x0.shape} and {x1.shape}!'
            )

        p0, z1 = self._forward(x0, x1)
        p1, z0 = self._forward(x1, x0)

        return (z0, p0), (z1, p1)

class MomentumBT(BenchmarkModule):
    def __init__(self, config, dataloader_kNN, gpus):
        super().__init__(config, dataloader_kNN, gpus)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        device = 'cuda' if gpus else 'cpu'
        self.config = config
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        self.resnet_mmt_bt = \
            AddMomentum(self.backbone, num_ftrs=512, num_mlp_layers=3)
        self.criterion = BarlowTwinsLoss(device=device)
            
    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        (z0, p0), (z1, p1) = self.resnet_mmt_bt(x0, x1)
        # Symmetrized loss function
        loss = self.criterion(p0, z1) / 2 + self.criterion(p1, z0) / 2
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
        optim = torch.optim.SGD(self.resnet_mmt_bt.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.config['max_epochs'])
        return [optim], [scheduler]

    def get_view(self):
        id1, id2 = sample([0,1,2,3], 2)
        view1_idx = range(id1 * 3, id1 * 3 + 3)
        view2_idx = range(id2 * 3, id2 * 3 + 3)
        return view1_idx, view2_idx

class VanillaBT(BenchmarkModule):
    def __init__(self, config, dataloader_kNN, gpus):
        super().__init__(config, dataloader_kNN)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        device = 'cuda' if gpus else 'cpu'
        self.config = config
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simsiam model based on ResNet
        # note that bartontwins has the same architecture
        self.resnet_simsiam = \
            lightly.models.SimSiam(self.backbone, num_ftrs=512, num_mlp_layers=3)
        self.criterion = BarlowTwinsLoss(device=device)
            
    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simsiam(x0, x1)
        # our simsiam model returns both (features + projection head)
        z_a, _ = x0
        z_b, _ = x1
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
