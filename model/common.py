import torch
import torch.nn as nn


class Conv(nn.Module):
    """
    base C B R block
    input shape : b c1 w h
    output shape :b c2 w h
    # 只改变channel大小，不改变shape形状。
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self.auto_pad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.relu = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    @staticmethod
    def auto_pad(k, p=None):
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p


class BaseConv(nn.Module):
    """
    default input shape same as out shape
    """

    def __init__(self, c1, c2, k=1, s=1, groups=1, bias=False, act="silu"):
        super(BaseConv, self).__init__()
        # same padding
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.rel = self.__get_activation(act)

    def forward(self, x):
        return self.rel(self.bn(self.conv(x)))

    @staticmethod
    def __get_activation(name="silu", inplace=True):
        if name == "silu":
            module = nn.SiLU(inplace=inplace)
        elif name == "relu":
            module = nn.ReLU(inplace=inplace)
        elif name == "lrelu":
            module = nn.LeakyReLU(0.1, inplace=inplace)

        else:
            raise AttributeError("Unsupported act type: {}".format(name))
        return module


class Focus(nn.Module):
    """Focus width and height information into channel space."""
    """
    shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.conv = BaseConv(c1 * 4, c2)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left,
                       patch_top_right, patch_bot_right),
                      dim=1)
        return self.conv(x)


class SeparableConv2d(nn.Module):
    """
    深度可分离卷积：
     （1）作用：backbone 做特征提取。优点：参数数量和运算成本比较低.
     （2）实现：有两部分组成。
            （2.1）depthwise(DW)卷积，通过指定 groups = in_channels实现。
            （2.2）pointwise(PW)点卷积 进行通道扩展。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding,
                               dilation=dilation, groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Bottleneck(nn.Module):
    """
    # base 残差
    """

    def __init__(self, c1, c2, shortcut=True, expansion=0.5, depthwise=False):
        super(Bottleneck, self).__init__()
        hidden_channels = int(c2 * expansion)
        Conv = DepthConv if depthwise else BaseConv
        self.conv1 = BaseConv(c1, hidden_channels, 1)
        self.conv2 = Conv(hidden_channels, c2, 3)
        self.use_add = shortcut and c1 == c2

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super(ResLayer, self).__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


# Spatial pyramid pooling layer        : 空间特征层池化操作，“不同尺度上的特征进行融合的”
# Atrous Spatial pyramid pooling layer : 空洞卷积的空间特征层池化操作，

class SPP(nn.Module):
    # Spatial pyramid pooling 融合“卷积上”不同尺度的特征
    # 空间赤化特征的
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class DilationConv(nn.Module):
    def __init__(self, c1, c2, k, d):
        # kernel_size and dilation keep Parameter as map sample:
        # kernel_size = [1,3,3,3]
        # dilation =    [1,6,12,18] or [1,12,18,36]
        super(DilationConv, self).__init__()

        padding = 0 if k == 1 else d
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, padding=padding, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ASPP(nn.Module):
    # Atrous Spatial pyramid pooling layer。 融合“空洞上”不同尺度的特征
    def __init__(self, c1, c2, out_stride):
        super(ASPP, self).__init__()
        kernel_size = [1, 3, 3, 3]
        dilation = [1, 6, 12, 18] if out_stride == 16 else [1, 12, 18, 36]
        self.assps = nn.ModuleList(DilationConv(c1, c2, k, d) for k, d in zip(kernel_size, dilation))
        self.conv = Conv(c2 * (len(dilation)), c2)

    def forward(self, x):
        # include ' x + ' means 残差。
        return self.conv(torch.cat([x] + [assp(x) for assp in self.assps], dim=1))


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP， YOLOv5"""
    """ SPP with Bottleneck"""

    def __init__(self, c1, c2, kernel_sizes=(5, 9, 13), activation="silu"):
        super(SPPBottleneck, self).__init__()
        c1_ = c1 // 2
        self.conv1 = BaseConv(c1, c1_, act=activation)
        # SPP 不同的 kernel_sizes 进行操作，padding 为卷积核大小的一半，输出大小相等，最后拼接
        #
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                                for ks in kernel_sizes])
        c_ = c1_ * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(c_, c2, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""
    """CSP 相对于 Res 结构的区别在于 CSP 是 source 与 source 卷积之后的特征进行，
        “拼接成两块”，然后在进行卷积操作，形成输入。 """
    """Res 相对于 CSP 结构的区别在于 Res 是 source 与 source 卷积之后的特征进行，
        “相加操作”，形成输入。 """

    def __init__(self, c1, c2, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu"):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(CSPLayer, self).__init__()
        hidden_channels = int(c2 * expansion)  # hidden channels
        self.conv1 = BaseConv(c1, hidden_channels)
        self.conv2 = BaseConv(c1, hidden_channels)
        self.conv3 = BaseConv(2 * hidden_channels, c2)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise)
                       for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


if __name__ == '__main__':

    data = torch.rand(10, 64, 256, 256)
    con = CSPLayer(64, 128, n=3)
    out = con(data)
    print(out.shape)
