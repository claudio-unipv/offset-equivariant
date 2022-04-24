#!/usr/bin/env python3

import argparse
import torch
import imgdata


"""
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
"""


def parse_args():
    parser = argparse.ArgumentParser("Eval a model on CIFAR-10 images")
    a = parser.add_argument
    a("model", help="Model file")
    a("-b", "--batch-size", type=int, default=128, help="Images per minibatch")
    a("-r", "--data-dir", default="./data", help="Directory for the storage of images")
    a("-w", "--workers", type=int, default=2, help="Parallel workers")
    dev = ("cuda" if torch.cuda.is_available() else "cpu")
    a("-d", "--device", default=dev, help="Computing device")
    a("-i", "--illuminant-saturation", type=float, default=0.0,
      help="Saturation of random illuminants")
    a("-e", "--equivariant", action="store_true", help="Use the equivariant version of the model")
    a("--color-constancy", action="store_true", help="Use cc preprocessing")
    return parser.parse_args()


def denormalize(x):
    m = torch.tensor(imgdata.CIFAR10_MEAN, device=x.device).view(3, 1, 1)
    s = torch.tensor(imgdata.CIFAR10_STD, device=x.device).view(3, 1, 1)
    return torch.clamp((s * x) + m, 0, 1)


def normalize(x):
    m = torch.tensor(imgdata.CIFAR10_MEAN, device=x.device).view(3, 1, 1)
    s = torch.tensor(imgdata.CIFAR10_STD, device=x.device).view(3, 1, 1)
    return (x - m) / s


def test_epoch(net, loader, device, qucc):
    # qucc: use color constancy preprocessing model
    if qucc:
        import qucc
        import torchvision
        quccnet = qucc.QUCC(device)
    n = 0
    tot_error = 0.0
    net.eval()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            if qucc:
                images = normalize(quccnet.process(denormalize(images)))
            outputs = net(images)
        tot_error += (outputs.argmax(1) != labels).float().sum().item()
        n += outputs.size(0)
    return tot_error / n


def main():
    args = parse_args()

    # Load the model
    data = torch.load(args.model, map_location=args.device)
    net = imgdata.make_model(data["args"].depth, args.equivariant)
    net.load_state_dict(data["model_state_dict"])
    net.to(args.device)

    # Setup the data
    loader = imgdata.make_cifar_loader(False, args, args.illuminant_saturation,
                                       args.equivariant)

    # Evaluation on the test set
    error = test_epoch(net, loader, args.device, args.color_constancy)
    print(f"Error: {error:.4f}")


if __name__ == "__main__":
    main()
