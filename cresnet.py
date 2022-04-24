"""Resnet for CIFAR 10.

Implementation of the models described in:

   CVPR 2016, Deep Residual Learning for Image Recognition, by Kaiming
   He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

With respect to the standard models for Image NET, those for CIFAR 10
are simpler, and have fewer parameters.

"""

import torch
import torch.nn.functional as F


class CResnet(torch.nn.Module):
    """Resnet model for CIFAR 10.

    Args:
        classes (int): number of output classes
        depth (int): parameter n in the paper (the actual depth is 2 * len(features) + 2)
        channels (int): number of input channels (e.g. 3 for color images)
        features (list[int]): channels in the feature maps

    """

    def __init__(self, classes, depth, channels=3, features=[16, 32, 64]):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, features[0], 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(features[0])
        self.blocks = self._make_blocks(depth, features)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(features[-1], classes)

    def _make_blocks(self, depth, features):
        # f[0], ..., f[0], f[1], ..., f[1], f[2], ...
        # the first is repeated one more time since is the size of input features
        channels = [features[0]] + [n for n in features for _ in range(depth)]
        ms = [ResidualBlock(in_, out, 1 + (in_ != out)) for in_, out in zip(channels, channels[1:])]
        return torch.nn.Sequential(*ms)

    def features(self, x):
        """Extract local features from a batch of images."""
        x = F.relu(self.bn(self.conv(x)))
        x = self.blocks(x)
        return x

    def forward(self, x):
        """Compute logits for the batch of images x."""
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResidualBlock(torch.nn.Module):
    """Residual block performin two convolutions with a skip connection.

    The module compute (omitting batch normalization):
        y = relu(x + conv2(relu(conv1(x))))
    where before the addition x is padded and downsampled if needed.

    Args:
        input_channels (int): number of channels in input maps
        output_channels (int): number of channels in output maps
        stride (int): stride applied in the first convolution
    """

    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channels)
        self.conv2 = torch.nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channels)
        self.stride = stride
        if input_channels != output_channels:
            self.padding = (0, 0, 0, 0, 0, output_channels - input_channels)
        else:
            self.padding = None

    def forward(self, x):
        """Apply the block to x."""
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.padding is not None:
            x = F.pad(x, self.padding)
        return F.relu(x + y)


def _test():
    depth = 3
    net = CResnet(10, depth)
    print(net)
    x = torch.zeros(2, 3, 32, 32)
    f = net.features(x)
    y = net(x)
    cs = sum(1 for m in net.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)))
    ps = sum(p.numel() for p in net.parameters())

    def sz(x):
        return "x".join(map(str, x.size()))

    print("Depth", cs)
    print(f"{ps / 1000000:.2f}M ({ps}) parameters")
    print(sz(x), "->", sz(f), "->", sz(y))


if __name__ == "__main__":
    _test()
