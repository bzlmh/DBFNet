import torch
import torch.nn as nn
from models.dfdb.GateConv import GateConv, GateDeConv
from models.dfdb.Supervisory import Supervisory
from models.dfdb.GateConv import _DenseLayer
from models.dfdb.GateConv import _Transition
import sys
class BiFPN_Add2(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add2, self).__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1]))
from einops.layers.torch import Rearrange
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, same_shape=True, **kwargs):
        super(Residual, self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = GateConv(in_channels, 2 * in_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = GateConv(in_channels, out_channels, kernel_size=3, padding=1)
        if not same_shape:
            self.conv3 = GateConv(in_channels, out_channels, kernel_size=1,
                                  stride=strides)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = out + x
        return F.relu(out)
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 添加池化层

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.pool(x)  # 在卷积操作后应用池化操作
        return x




class GFBNet(nn.Module):
    """
    Generator Using Gate Convolution
    """

    def upsamplelike(self, inputs):
        src, target = inputs
        # 如果通道数不匹配，则调整通道数
        if src.shape[1] != target.shape[1]:
            src = self.adjust_channels(src, target)
        # 执行上采样操作
        return F.interpolate(src, size=(target.shape[2], target.shape[3]), mode='nearest')

    def adjust_channels(self, src, target):
        src_channels = src.shape[1]
        target_channels = target.shape[1]
        # 使用卷积层调整通道数
        conv = nn.Conv2d(src_channels, target_channels, kernel_size=1).to('cuda')
        adjusted_src = conv(src).to('cuda')
        return adjusted_src

    def __init__(self, input_c):
        super(GFBNet, self).__init__()
        self.c = 64  # 设定特征通道数
        #门控卷积
        self.prj_5 = nn.Conv2d(256, 256, kernel_size=1)
        self.prj_4 = nn.Conv2d(128, 128, kernel_size=1)
        self.prj_3 = nn.Conv2d(64, 64, kernel_size=1)
        self.prj_2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv_smooth = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU())
        self.corase_a1_ = _DenseLayer(num_input_features=5, growth_rate=128, bn_size=4, drop_rate=0.1)
        self.transition_layer1 = _Transition(num_input_feature=133, num_output_features=64, stride=1, kernel_size=1)
        self.sa1 = SpatialAttention()
        self.ca1 = ChannelAttention(64,reduction=8)
        self.pa1 = PixelAttention(64)
        self.corase_a1 = _DenseLayer(num_input_features=64, growth_rate=32, bn_size=4, drop_rate=0.1)
        self.transition_layer2 = _Transition(num_input_feature=96, num_output_features=64, stride=2, kernel_size=2)
        self.sa2 = SpatialAttention()
        self.ca2 = ChannelAttention(64,reduction=8)
        self.pa2 = PixelAttention(64)
        self.corase_a2 = _DenseLayer(num_input_features=64, growth_rate=32, bn_size=4, drop_rate=0.1)
        self.transition_layer3 = _Transition(num_input_feature=96, num_output_features=128, stride=1, kernel_size=1)
        self.sa3 = SpatialAttention()
        self.ca3 = ChannelAttention(128,reduction=8)
        self.pa3 = PixelAttention(128)
        self.corase_a3 = _DenseLayer(num_input_features=128, growth_rate=32, bn_size=4, drop_rate=0.1)
        self.transition_layer4 = _Transition(num_input_feature=160, num_output_features=64, stride=1, kernel_size=1)
        self.sa4 = SpatialAttention()
        self.ca4 = ChannelAttention(64,reduction=8)
        self.pa4 = PixelAttention(64)
        self.bifpn1 = BiFPN_Add2(256, 256)
        self.bifpn2 = BiFPN_Add2(128,128)
        self.bifpn3 = BiFPN_Add2(64, 64)
        self.bifpn4 = BiFPN_Add2(64, 64)
        # 门控卷积残差连接
        self.res1 = Residual(self.c, 2 * self.c, same_shape=False)
        self.res2 = Residual(self.c, 2 * self.c)
        self.res3 = Residual(self.c, 4 * self.c, same_shape=False)
        self.res4 = Residual(2 * self.c, 4 * self.c)
        self.res5 = Residual(2 * self.c, 8 * self.c, same_shape=False)
        self.res6 = Residual(4 * self.c, 8 * self.c)
        self.res7 = Residual(4 * self.c, 16 * self.c, same_shape=False)
        self.res8 = Residual(8 * self.c, 16 * self.c)
        self.hidelayer1 = Supervisory(in_channels=256, out_channel=1, scale_factor=16)
        self.hidelayer2 = Supervisory(in_channels=128, out_channel=1, scale_factor=8)
        self.hidelayer3 = Supervisory(in_channels=64, out_channel=1, scale_factor=4)
        self.hidelayer4 = Supervisory(in_channels=64, out_channel=1, scale_factor=2)
        self.gateDeConv = GateDeConv(self.c, 2, kernel_size=3, stride=1, padding=1, activation=torch.sigmoid)
    def forward(self, ori, ostu, sobel):
        #门控卷积
        img_input = torch.cat((ori, ostu, sobel), 1)  # 在第一个维度上拼接，添加批处理维度
        y = self.corase_a1_(img_input)
        y = self.transition_layer1(y)
        cattn = self.ca1(y)
        sattn = self.sa1(y)
        pattn1 = sattn + cattn
        pattn2 = self.pa1(y, pattn1)
        y = pattn2 * y
        y = self.corase_a1(y)
        y = self.transition_layer2(y)
        cattn = self.ca2(y)
        sattn = self.sa2(y)
        pattn1 = sattn + cattn
        pattn2 = self.pa2(y, pattn1)
        y = pattn2 * y
        y = self.corase_a2(y)
        y = self.transition_layer3(y)
        cattn = self.ca3(y)
        sattn = self.sa3(y)
        pattn1 = sattn + cattn
        pattn2 = self.pa3(y, pattn1)
        y = pattn2 * y
        y = self.corase_a3(y)
        y = self.transition_layer4(y)
        cattn = self.ca4(y)
        sattn = self.sa4(y)
        pattn1 = sattn + cattn
        pattn2 = self.pa4(y, pattn1)
        y = pattn2 * y
        # skip_connections.append(y)
        C2 = self.prj_2(y)
        y = self.res1(y)
        y = self.res2(y)
        # skip_connections.append(y)
        C3 = self.prj_3(y)  # 卷积一次
        y = self.res3(y)
        y = self.res4(y)
        # skip_connections.append(y)
        C4 = self.prj_4(y)  # 卷积一次
        y = self.res5(y)
        y = self.res6(y)
        # skip_connections.append(y)
        C5 = self.prj_5(y)  # 卷积一次
        y = self.res7(y)
        y = self.res8(y)
        # #融合张量
        P5= self.bifpn1([C5, self.upsamplelike([y, C5])])

        mid_outP5 = self.hidelayer1(P5)
        P4 = self.bifpn2([C4,self.upsamplelike([P5, C4])])
        mid_outP4= self.hidelayer2(P4)
        P3 = self.bifpn3([C3, self.upsamplelike([P4,C3])])
        mid_outP3 = self.hidelayer3(P3)
        P2 = self.bifpn4([C2 , self.upsamplelike([P3, C2])])
        mid_outP2 = self.hidelayer4(P2)
        P2=self.conv_smooth(P2)
        bin_out=self.gateDeConv(P2)
        return bin_out,mid_outP2,mid_outP3,mid_outP5,mid_outP4

# test = GFBNet(input_c=5)
# ori = torch.ones((1, 3,256, 256))
# ostu = torch.ones((1, 1, 256, 256))
# sobel = torch.ones((1, 1,256, 256))
# # 执行前向传播
# output = test(ori, ostu, sobel)
