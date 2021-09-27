import os
import torch
import torch.nn as nn
import torch.nn.functional as F


import sys
import torch
from collections import OrderedDict
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, bn_momentum=0.0003):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
    
    def _sum_each(self, x, y):
        assert(len(x)==len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_class, block, layers, dilation=[1,1,1,1], bn_momentum=0.0003, is_fpn=False):
        self.inplanes = 128
        self.is_fpn = is_fpn
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_class, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=bn_momentum)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn3 = BatchNorm2d(128, momentum=bn_momentum)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=dilation[0], bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1 if dilation[1]!=1 else 2, dilation=dilation[1], bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1 if dilation[2]!=1 else 2, dilation=dilation[2], bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1 if dilation[3]!=1 else 2, dilation=dilation[3], bn_momentum=bn_momentum)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, bn_momentum=0.0003):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine = True, momentum=bn_momentum))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid), bn_momentum=bn_momentum))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid), bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def forward(self, x, start_module=1, end_module=5):
        if start_module <= 1:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.maxpool(x)
            start_module = 2
        features = []
        for i in range(start_module, end_module+1):
            x = eval('self.layer%d'%(i-1))(x)
            features.append(x)

        if self.is_fpn:
            if len(features) == 1:
                return features[0]
            else:
                return tuple(features)
        else:
            return x


def get_resnet101(in_class, dilation=[1,1,1,1], bn_momentum=0.0003, is_fpn=False):
    model = ResNet(in_class,Bottleneck, [3, 4, 23, 3], dilation=dilation, bn_momentum=bn_momentum, is_fpn=is_fpn)
    return model



class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)       # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool


class DeepLabV3(nn.Module):
    def __init__(self, in_class, class_num, bn_momentum=0.01):
        super(DeepLabV3, self).__init__()
        self.Resnet101 = get_resnet101(in_class, dilation=[1, 1, 1, 2], bn_momentum=bn_momentum, is_fpn=False)
        self.ASPP = ASPP(2048, 256, [6, 12, 18], norm_act=nn.BatchNorm2d)
        self.classify = nn.Conv2d(256, class_num, 1, bias=True)
        
    def forward(self, input):
        x = self.Resnet101(input)

        aspp = self.ASPP(x)     # 空间金字塔池化
        predict = self.classify(aspp)

        output= F.interpolate(predict, size=input.size()[2:4], mode='bilinear', align_corners=True)
        return output

