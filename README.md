# Offset equivariant networks

Pytorch implementation of offset equivariant networks.

Offset equivariant networks have a predictable behavior in face of uniform changes of the input values.
In the log RGB space thsse networks are equivariant to global changes in the illumination.
This makes it easy to achieve, for instance, image recognition invariant wrt the color of the light source.

## Setup

First, clone the repository and set the working directory.
Then, setup a virtual environment and install the required libraries.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements
```

The project has been tested with python 3.8.10, pytorch 1.11.0, torchvision 0.12.0 and CUDA 11.4.


## Train the models

Train a standard resnet model on CIFAR images:
```
python3 train_cifar.py -S standard.pth
```

The '-e' switch makes it so the equivariant version is trained:
```
python3 train_cifar.py -e -S equivariant.pth
```

## Evaluate the models

```
python3 eval_cifar.py standard.pth
python3 eval_cifar.py -e equivariant.pth
```
Pay attention to use the '-e' switch with the equivariant version.
