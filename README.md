# Exploring Asymmetry in Barlow Twins

PyTorch Implementation of Barlow Twins paper: [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/pdf/2103.03230.pdf)

This is work done under MLO Laboratory, EPFL as a Summer@EPFL intern. The code is a modified version of the Barlow Twins implementation [here](https://github.com/IgorSusmelj/barlowtwins) 

### Installation

`pip install -r requirements.txt`

### Dependencies

- PyTorch
- PyTorch Lightning
- Torchvision
- lightly
- Wandb

### Benchmarks
We benchmark the BarlowTwins model on the CIFAR-10 dataset following the KNN evaluation protocol. Currently, the best effort achieved a test accuracy of 84.7%.

Accuracy             |  Loss 
:-------------------------:|:-------------------------:
![](docs/accuracy_logs.png)  |  ![](docs/loss_logs.png)


### Paper

[Project Report](https://arxiv.org/pdf/2103.03230.pdf)
