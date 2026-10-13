import torch.nn as nn


def conv_bn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


class Hourglass(nn.Module):
    def __init__(self, channels, activate):
        super().__init__()
        self.conv1 = nn.Sequential(conv_bn_3d(channels, channels * 2, 3, 2, 1), activate)
        self.conv2 = conv_bn_3d(channels * 2, channels * 2, 3, 1, 1)
        self.conv3 = nn.Sequential(conv_bn_3d(channels * 2, channels * 2, 3, 2, 1), activate)
        self.conv4 = nn.Sequential(conv_bn_3d(channels * 2, channels * 2, 3, 1, 1), activate)
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels * 2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(channels * 2))
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(channels))
        self.activate = nn.Sequential(activate)

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)
        pre = self.conv2(out)
        pre = self.activate(pre if postsqu is None else pre + postsqu)
        out = self.conv4(self.conv3(pre))
        post = self.activate(self.conv5(out) + (pre if presqu is None else presqu))
        return self.conv6(post), pre, post


class S3DCNN(nn.Module):
    """GaussianOcc 的 3D 聚合。last_free 时语义头直接输出全部类别。"""

    def __init__(self, in_channels, num_classes, mid_channels=32):
        super().__init__()
        act = nn.ReLU(inplace=True)
        self.stem = nn.Sequential(
            conv_bn_3d(in_channels, mid_channels, 3, 1, 1), act,
            conv_bn_3d(mid_channels, mid_channels, 3, 1, 1), act)
        self.res = nn.Sequential(
            conv_bn_3d(mid_channels, mid_channels, 3, 1, 1), act,
            conv_bn_3d(mid_channels, mid_channels, 3, 1, 1))
        self.hourglasses = nn.ModuleList([Hourglass(mid_channels, act) for _ in range(3)])
        self.semantic = nn.Sequential(
            conv_bn_3d(mid_channels, mid_channels, 3, 1, 1), act,
            conv_bn_3d(mid_channels, mid_channels, 3, 1, 1), act,
            conv_bn_3d(mid_channels, mid_channels, 3, 1, 1), act,
            conv_bn_3d(mid_channels, mid_channels, 3, 1, 1), act,
            conv_bn_3d(mid_channels, num_classes, 3, 1, 1))

    def forward(self, cost):
        cost0 = self.stem(cost)
        cost0 = self.res(cost0) + cost0
        pre1 = post1 = None
        out = cost0
        for index, hourglass in enumerate(self.hourglasses):
            if index == 0:
                out, pre1, post1 = hourglass(out, None, None)
                out = out + cost0
            elif index == 1:
                out, _, post2 = hourglass(out, pre1, post1)
                out = out + cost0
            else:
                out, _, _ = hourglass(out, pre1, post2)
        return self.semantic(out)
