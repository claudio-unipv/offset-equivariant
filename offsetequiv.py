import torch


def add_offset(x, offset):
    """Add an offset to a tensor.

    Args:
        x (tensor): input tensor [N, C, ...]
        offset (tensor): offset [N, G]

    Returns:
        y (tensor): output tensor (same shape of x).

    The number of items in x must be of multiple of the length of the
    offset.

    """
    g = offset.size(1)
    y = x.view(x.size(0), g, -1) + offset.unsqueeze(-1)
    return y.view_as(x)


# Equivariant layers.  The are supposed to be drop-in replacements for
# standard layers.


class EquivariantBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, groups, features, *args, **kwargs):
        super().__init__(features, *args, **kwargs)
        self.features = features
        self.groups = groups

    def forward(self, x):
        y = x.view(x.size(0), self.groups, -1)
        m = y.mean(2, True)
        x = super().forward((y - m).view_as(x))
        return (x.view_as(y) + m).view_as(x)


class EquivariantLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, groups, bias=True, device=None, dtype=None):
        if in_features % groups != 0 or out_features % groups != 0:
            raise ValueError("The number of groups must divide the number of features")
        super().__init__()
        m = in_features
        n = out_features
        g = groups
        a = torch.div(g * torch.arange(m, device=device), m, rounding_mode="floor")
        b = torch.div(g * torch.arange(n, device=device), n, rounding_mode="floor")
        W0 = (a[None, :] == b[:, None]) * (g / m)
        self.register_buffer("W0", W0.to(dtype=dtype))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(n, dtype=dtype, device=device))
        else:
            self.bias = None
        # w = torch.empty(n, g, (m - g) // g, dtype=dtype, device=device)
        # w = torch.empty(n, m - g, dtype=dtype, device=device)
        w = torch.empty(n, g, m // g, dtype=dtype, device=device)
        torch.nn.init.xavier_uniform_(w)
        self.weight = torch.nn.Parameter(w)
        # G = (a[:, None] == torch.arange(g, device=device)[None, :]).float()
        # N = torch.linalg.svd(G)[0]
        # N = (N.T)[g:]
        # self.register_buffer("N", N.to(dtype=dtype))

    def forward(self, x):
        # U = self.weight
        # W = torch.cat((-U.sum(2, True), U), 2).view_as(self.W0)
        # W = torch.cat((U[:, :, :1], U[:, :, 1:] - U[:, :, :-1], -U[:, :, -1:]), 2)
        # W = self.weight @ self.N
        W = self.weight - self.weight.mean(2, True)
        W = W.view_as(self.W0) + self.W0
        return torch.nn.functional.linear(x, W, self.bias)

    def __repr__(self):
        n, m = self.W0.size()
        g = self.weight.size(1)
        b = (self.bias is not None)
        return f"{type(self).__name__}({m}, {n}, {g}, bias={b})"


class EquivariantConv2d(torch.nn.Module):
    def __init__(self, in_features, out_features, groups, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, device=None, dtype=None):
        if in_features % groups != 0 or out_features % groups != 0:
            raise ValueError("The number of groups must divide the number of features")
        super().__init__()
        self.stride = stride
        self.dilation = dilation
        self.pad = torch.nn.ReplicationPad2d(padding)
        m = in_features
        n = out_features
        g = groups
        k = m * kernel_size * kernel_size
        a = torch.div(g * torch.arange(k, device=device), k, rounding_mode="floor")
        b = torch.div(g * torch.arange(n, device=device), n, rounding_mode="floor")
        W0 = (a[None, :] == b[:, None]) * (g / k)
        W0 = W0.view(n, m, kernel_size, kernel_size)
        self.register_buffer("W0", W0.to(dtype=dtype))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(n, dtype=dtype, device=device))
        else:
            self.bias = None
        # w = torch.empty(n, g, k // g - 1, dtype=dtype, device=device)
        # w = torch.empty(n, k - g, dtype=dtype, device=device)
        w = torch.empty(n, g, k // g, dtype=dtype, device=device)
        torch.nn.init.xavier_uniform_(w)
        self.weight = torch.nn.Parameter(w)
        # G = (a[:, None] == torch.arange(g, device=device)[None, :]).float()
        # N = torch.linalg.svd(G)[0]
        # N = (N.T)[g:]
        # self.register_buffer("N", N.to(dtype=dtype))

    def forward(self, x):
        # U = self.weight
        # p = (self.W0.size(2) * self.W0.size(3) - 1) // 2
        # W = torch.cat((U[:, :, :p], -U.sum(2, True), U[:, :, p:]), 2)
        # W = torch.cat((U[:, :, :1], U[:, :, 1:] - U[:, :, :-1], -U[:, :, -1:]), 2)
        # W = self.weight @ self.N
        W = self.weight - self.weight.mean(2, True)
        W = W.view_as(self.W0) + self.W0
        x = self.pad(x)
        y = torch.nn.functional.conv2d(x, W, self.bias, stride=self.stride, dilation=self.dilation)
        return y

    def __repr__(self):
        n, m, k, _ = self.W0.size()
        g = self.weight.size(1)
        s = self.stride
        d = self.dilation
        p = self.pad.padding[0]
        b = (self.bias is not None)
        name = type(self).__name__
        return f"{name}({m}, {n}, {g}, {k}, stride={s}, padding={p}, dilation={d}, bias={b})"


class EquivariantReLU(torch.nn.Module):
    def __init__(self, groups):
        self.groups = groups
        super().__init__()

    def forward(self, x):
        if x.ndim != 4:
            y = x.unsqueeze(-1).unsqueeze(-1)
        else:
            y = x
        y = x.view(x.size(0), self.groups, -1)
        m = y.mean(2, True)
        y = torch.maximum(y, m)
        return y.view_as(x)

    def __repr__(self):
        return f"{type(self).__name__}({self.groups})"


class EquivariantReLU2(torch.nn.Module):
    def __init__(self, groups, features):
        self.groups = groups
        self.features = features
        super().__init__()
        self.linear = EquivariantConv2d(self.features, self.groups, self.groups, 1, bias=False)

    def forward(self, x):
        if x.ndim != 4:
            y = x.unsqueeze(-1).unsqueeze(-1)
        else:
            y = x
        m = self.linear(y).unsqueeze(2)
        y = y.view(y.size(0), self.groups, -1, y.size(2), y.size(3))
        y = torch.maximum(y, m)
        return y.view_as(x)

    def __repr__(self):
        return f"{type(self).__name__}({self.groups})"


# Adapters turn a regular module into an offset equivariant one by
# using group pooling functions:
#
#    h(x) = f(x - Gp(x)) + Gp(x)

class AvgAdapter(torch.nn.Module):
    def __init__(self, groups, operation):
        super().__init__()
        self.groups = groups
        self.operation = operation

    def forward(self, x):
        m = x.view(x.size(0), self.groups, -1).mean(2)
        y = self.operation(add_offset(x, -m))
        return add_offset(y, m)


class AvgPool2dAdapter(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        m = self.pool(x)
        return self.operation(x - m) + m


class LinearAdapter(torch.nn.Module):
    def __init__(self, groups, channels, operation):
        super().__init__()
        if channels % groups != 0:
            raise ValueError("The number of channels must be multiple of the number of groups")
        self.groups = groups
        self.operation = operation
        w = torch.empty(groups, channels // groups)
        torch.nn.init.xavier_uniform_(w)
        self.weight = torch.nn.Parameter(w)

    def forward(self, x):
        xm = (x if x.dim() == 2 else x.view(x.size(0), x.size(1), -1).mean(-1))
        xm = xm.view(xm.size(0), self.groups, -1)
        adj_w = 1 / self.weight.size(1) - self.weight.mean(1, True)
        w = self.weight + adj_w
        mu = torch.einsum("bgc, gc -> bg", xm, w)
        y = self.operation(add_offset(x, -mu))
        return add_offset(y, mu)


class BiLinearAdapter(torch.nn.Module):
    def __init__(self, groups, channels, operation):
        super().__init__()
        if channels % groups != 0:
            raise ValueError("The number of channels must be multiple of the number of groups")
        self.groups = groups
        self.operation = operation
        w = torch.empty(groups, channels // groups)
        torch.nn.init.xavier_uniform_(w)
        self.weight1 = torch.nn.Parameter(w)
        w = torch.empty(groups, channels // groups)
        torch.nn.init.xavier_uniform_(w)
        self.weight2 = torch.nn.Parameter(w)

    def forward(self, x):
        xm = (x if x.dim() == 2 else x.view(x.size(0), x.size(1), -1).mean(-1))
        xm = xm.view(xm.size(0), self.groups, -1)
        adj_w = 1 / self.weight1.size(1) - self.weight1.mean(1, True)
        w = self.weight1 + adj_w
        mu1 = torch.einsum("bgc, gc -> bg", xm, w)
        adj_w = 1 / self.weight2.size(1) - self.weight2.mean(1, True)
        w = self.weight2 + adj_w
        mu2 = torch.einsum("bgc, gc -> bg", xm, w)
        y = self.operation(add_offset(x, -mu1))
        return add_offset(y, mu2)


# The projection can be used to make a standard nn.Linear or nn.Conv2d
# layer into an equivariant one.  Just call the
# make_linear_equivariant function.


class _WeightProjection(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, weight):
        n = weight.size(0)
        m = weight.size(1)
        c = self.groups
        w = weight.view(c, n // c, c, m // c, -1)
        w = w - w.mean(3, keepdim=True)
        w = w + torch.eye(c, device=weight.device).view(c, 1, c, 1, 1) * c / (m * w.size(-1))
        return w.view_as(weight)

    def right_inverse(self, weight):
        return weight


def make_linear_equivariant(linear_or_conv, groups):
    p = _WeightProjection(groups)
    torch.nn.utils.parametrize.register_parametrization(linear_or_conv, "weight", p)
    if hasattr(linear_or_conv, "padding_mode"):
        if linear_or_conv.padding_mode not in ("replicate", "reflect", "circular"):
            linear_or_conv.padding_mode = "replicate"


def _test2d():
    bsz = 2
    m = 4
    n = 2
    g = 2
    dtype = torch.float32
    x = torch.randn(bsz, m, 3, 3, dtype=dtype)
    net = torch.nn.Sequential(
        EquivariantConv2d(m, n, g, 3, padding=1, dtype=dtype),
        EquivariantBatchNorm2d(g, n, dtype=dtype),
        EquivariantReLU(g),
    )
    net.eval()
    off = torch.randn(bsz, g)
    y1 = add_offset(net(x), off)
    y2 = net(add_offset(x, off))
    print(off.view(-1), (y2 - y1).view(-1))
    if torch.allclose(y1, y2):
        print("OK")
    else:
        print("NOT OK")
        print("Max diff:", torch.abs(y2 - y1).max().item())
        breakpoint()


def _test():
    bsz = 10
    x = torch.randn(bsz, 15)
    net = torch.nn.Sequential(
        EquivariantLinear(15, 30, 3),
        EquivariantReLU(3),
        EquivariantLinear(30, 6, 3),
        EquivariantReLU(3),
    )
    # net = torch.nn.Sequential(
    #     torch.nn.Linear(15, 30),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(30, 6),
    #     torch.nn.ReLU()
    # )
    # net = LinearAdapter(3, 15, net)
    off = torch.randn(bsz, 3)
    y1 = add_offset(net(x), off)
    y2 = net(add_offset(x, off))
    if torch.allclose(y1, y2):
        print("OK")
    else:
        print("NOT OK")
        print("Max diff:", torch.abs(y2 - y1).max().item())
        breakpoint()


if __name__ == "__main__":
    _test()
    _test2d()
