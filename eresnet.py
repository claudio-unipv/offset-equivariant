import sys
import torch
import torch.nn.functional as F
sys.path.append("..")
import offsetequiv as oe


class EResnet(torch.nn.Module):
    """Offset-equivariant resnet."""
    def __init__(self, classes, depth, channels=3, features=[15, 33, 63]):
        super().__init__()
        self.conv = oe.EquivariantConv2d(channels, features[0], 3, 3, padding=1)
        self.bn = oe.EquivariantBatchNorm2d(3, features[0])
        self.relu = oe.EquivariantReLU(3)
        self.blocks = self._make_blocks(depth, features)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = oe.EquivariantLinear(features[-1], 3 * classes, 3)

    def _make_blocks(self, depth, features):
        # f[0], ..., f[0], f[1], ..., f[1], f[2], ...
        # the first is repeated one more time since is the size of input features
        channels = [features[0]] + [n for n in features for _ in range(depth)]
        ms = [ResidualBlock(in_, out, 1 + (in_ != out)) for in_, out in zip(channels, channels[1:])]
        return torch.nn.Sequential(*ms)

    def features(self, x):
        """Extract local features from a batch of images."""
        x = self.relu(self.bn(self.conv(x)))
        x = self.blocks(x)
        return x

    def forward(self, x):
        """Compute logits for the batch of images x."""
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(x.size(0), 3, -1).mean(1)
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
        self.conv1 = oe.EquivariantConv2d(input_channels, output_channels, 3, 3, stride=stride, padding=1)
        self.bn1 = oe.EquivariantBatchNorm2d(3, output_channels)
        self.relu1 = oe.EquivariantReLU(3)
        self.conv2 = oe.EquivariantConv2d(output_channels, output_channels, 3, 3, padding=1)
        self.bn2 = oe.EquivariantBatchNorm2d(3, output_channels)
        self.relu2 = oe.EquivariantReLU(3)
        self.stride = stride
        if input_channels != output_channels:
            self.padding = (0, 0, 0, 0, 0, output_channels - input_channels)
        else:
            self.padding = None

    def forward(self, x):
        """Apply the block to x."""
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        m = x.view(x.size(0), 3, -1).mean(2)
        # m must be removed otherwise the offset will be added twice.
        # This has to be done before padding, otherwise zero-padding
        # would not be offset equivariant.
        x = oe.add_offset(x, -m)
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.padding is not None:
            x = F.pad(x, self.padding)
        return self.relu2(x + y)


def _test():
    depth = 3
    net = EResnet(10, depth)
    print(net)
    x = torch.rand(2, 3, 32, 32)
    f = net.features(x)
    y = net(x)
    cs = sum(1 for m in net.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)))
    ps = sum(p.numel() for p in net.parameters())

    def sz(x):
        return "x".join(map(str, x.size()))

    print("Depth", cs)
    print(f"{ps / 1000000:.2f}M ({ps}) parameters")
    print(sz(x), "->", sz(f), "->", sz(y))

    print("Checking invariance...")
    off = torch.randn(x.size(0), 3)
    y1 = torch.softmax(net(x), 1)
    y2 = torch.softmax(net(oe.add_offset(x, off)), 1)
    if torch.allclose(y1, y2):
        print("OK")
    else:
        print("NOT OK")
        breakpoint()


if __name__ == "__main__":
    _test()
