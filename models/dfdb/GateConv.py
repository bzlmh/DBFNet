import torch
import torch.nn as nn
import torch.nn.functional as F


class GateConv(torch.nn.Module):
    """
    Gate Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2)):
        super(GateConv, self).__init__()
        self.gate01 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        #print('input', input.shape)
        x = self.gate01(input)
        if self.activation is None:
            return x
        x, y = torch.chunk(x, 2, 1)
        y = torch.sigmoid(y)
        x = self.activation(x)
        x = x * y
        #print('output', x.shape)
        return x



class GateDeConv(torch.nn.Module):
    """
    Gate Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2)):
        super(GateDeConv, self).__init__()
        self.gate01 = GateConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, activation=activation)

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2, mode='bilinear')
        x = self.gate01(x)
        return x
class _DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate)
        # 在通道维上将输入和输出连结
        return torch.cat([x, new_features], 1)
class _Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, num_input_feature, num_output_features,stride=1,kernel_size=1):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=kernel_size, stride=stride))
