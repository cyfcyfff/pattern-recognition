import torch
import torch.nn as nn
import torchvision
from torch.nn.modules.dropout import Dropout
import math

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, channel, stride)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channel, channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x                                                # 记录这一步的特征数据

        out = self.conv1(x)                                         # 3x3卷积
        out = self.bn1(out)                                         # BN层
        out = self.relu(out)                                        # relu

        out = self.conv2(out)                                       # 3x3卷积
        out = self.bn2(out)                                         # bn层

        if self.downsample is not None:                             # 下采样
            residual = self.downsample(x)

        out += residual                                             # 残差相加
        out = self.relu(out)                                        # relu

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(channel, channel * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channel * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)                                 # 1x1卷积
        out = self.bn1(out)                                 # bn层
        out = self.relu(out)                                # relu

        out = self.conv2(out)                               # 3x3卷积
        out = self.bn2(out)                                 # bn层
        out = self.relu(out)                                # relu

        out = self.conv3(out)                               # 1x1卷积
        out = self.bn3(out)

        if self.downsample is not None:                     # 下采样
            residual = self.downsample(x)

        out += residual                                     # 残差相加
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    #                  basic  [2,2,2,2]
    def __init__(self, block, layers, num_class1=4, num_class2=3):
        self.in_channel = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)               # 7x7卷积
        self.bn1 = nn.BatchNorm2d(64)                                                               # BN层
        self.relu = nn.ReLU(inplace=True)                                                           # relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                             # maxpool
        self.layer1 = self._make_layer(block, 64, layers[0])                                        # "layers[0]"个block
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)                                                    # average pool
        self.fc1 = nn.Linear(512 * block.expansion, num_class1)                                              # fc
        self.fc2 = nn.Linear(512 * block.expansion, num_class2)                                              # fc

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    #                     block  in_channels block数
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 步长不为1，输入维度 ！= planes * block.expansion
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 定义下采样函数
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)               # 7x7卷积
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)             # 最大池化

        x = self.layer1(x)              # layer
        x = self.layer2(x)              # layer
        x = self.layer3(x)              # layer
        x = self.layer4(x)              # layer

        x = self.avgpool(x)             # 平均池化


        x = x.view(x.size(0), -1)
        c1 = self.fc1(x)
        c2 = self.fc2(x)
        return c1, c2

def resnet18(pretrained=False, **kwargs):
    #              block,      layers
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)#对应图中的2x2x2x2

    return model