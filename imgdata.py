import os
import PIL
import torch
import torchvision
import ptcolor
import eresnet
import cresnet


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------


def make_model(depth, equivariant):
    if equivariant:
        net = eresnet.EResnet(10, depth, features=[15, 33, 63])
    else:
        net = cresnet.CResnet(10, depth, features=[16, 32, 64])
    return net


def random_illuminant(s):
    def transform(x):
        if s <= 0:
            return x
        hsv = torch.rand(3, 1, 1)
        hsv.data[1] = s
        hsv.data[2] = 1.0
        ill = ptcolor.hsv2rgb(hsv[None, ...])[0]
        return x * ill
    return transform


def apply_illuminant(h, s):
    def transform(x):
        if s <= 0:
            return x
        hsv = torch.tensor([h, s, 1.0])
        ill = ptcolor.hsv2rgb(hsv[None, :, None, None])[0]
        return x * ill
    return transform


def rgb2logrgb(rgb, eps=0.002):
    rgb = torch.clamp(rgb, eps)
    return -torch.log(rgb)


def logrgb2rgb(logrgb):
    return torch.exp(-logrgb)

# ------------------------------------------------------------
# CIFAR 10
# ------------------------------------------------------------


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_LOGRGB_MEAN = (1.6430, 1.6853, 1.8957)
CIFAR10_LOGRGB_STD = (1.4121, 1.4130, 1.5579)


def make_cifar_loader(train, args, ill_saturation, equivariant, illuminant_hue=None,
                      aug_sat=0.0, aug_hue=0.0, aug_bri=0.0, aug_con=0.0):
    trs = [torchvision.transforms.ToTensor(), ptcolor.remove_gamma]
    if illuminant_hue is None:
        trs.append(random_illuminant(ill_saturation))
    else:
        trs.append(apply_illuminant(illuminant_hue, ill_saturation))
    if not equivariant:
        trs.append(ptcolor.apply_gamma)
    if train and any([aug_sat, aug_hue, aug_bri, aug_con]):
        opts = {
            "brightness": aug_bri,
            "contrast": aug_con,
            "saturation": aug_sat,
            "hue": aug_hue
        }
        trs.append(torchvision.transforms.ColorJitter(**opts))
    if equivariant:
        trs.append(rgb2logrgb)
        trs.append(torchvision.transforms.Normalize(CIFAR10_LOGRGB_MEAN, CIFAR10_LOGRGB_STD))
    else:
        trs.append(torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    if train:
        trs.append(torchvision.transforms.RandomCrop(32, padding=4))
        trs.append(torchvision.transforms.RandomHorizontalFlip())

    transform = torchvision.transforms.Compose(trs)
    dset = torchvision.datasets.CIFAR10(root=args.data_dir, train=train,
                                        download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size,
                                         shuffle=train, num_workers=args.workers,
                                         drop_last=train)
    return loader




# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------


def compute_statistics(loader):
    """Compute mean and standard deviation."""
    tot = torch.zeros(3)
    n = 0
    for images, _ in loader:
        tot += images.sum(-1).sum(-1).sum(0).detach().cpu()
        n += images.size(0) * images.size(2) * images.size(3)
    mean = tot / n
    tot2 = torch.zeros(3)
    n = 0
    for images, _ in loader:
        diff = (images - mean.view(1, 3, 1, 1)) ** 2
        tot2 += diff.sum(-1).sum(-1).sum(0).detach().cpu()
        n += images.size(0) * images.size(2) * images.size(3)
    var = tot2 / n
    return mean, torch.sqrt(var)


def _test_cifar():
    from types import SimpleNamespace
    args = SimpleNamespace(workers=0, batch_size=128, data_dir="./data")
    loader = make_cifar_loader(True, args, 0.0, False)
    mu, sigma = compute_statistics(loader)
    print("MEAN ", *mu.tolist())
    print("SIGMA", *sigma.tolist())


if __name__ == "__main__":
    _test_cifar()
