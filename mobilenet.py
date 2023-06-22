import torch
import torch.nn as nn
import torchvision
from torch.nn.modules.dropout import Dropout


# 定义h-swith激活函数
class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x*self.relu6(x+3)/6

# DW卷积
def ConvBNActivation(in_channels,out_channels,kernel_size,stride,activate):
    # 通过设置padding达到当stride=2时，hw减半的效果。此时不与kernel_size有关，所实现的公式为: padding=(kernel_size-1)//2
    # 当kernel_size=3,padding=1时: stride=2 hw减半, stride=1 hw不变
    # 当kernel_size=5,padding=2时: stride=2 hw减半, stride=1 hw不变
    # 从而达到了使用 stride 来控制hw的效果， 不用去关心kernel_size的大小，控制单一变量
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish()
        )

# PW卷积(接全连接层)
def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

# 普通的1x1卷积
def Conv1x1BNActivation(in_channels,out_channels,activate):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish()
        )

# 注意力机制(SE模块)
class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels, se_kernel_size, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide   # 维度变为原来的1/4

        # 将当前的channel平均池化成1
        self.pool = nn.AvgPool2d(kernel_size=se_kernel_size, stride=1)

        # 两个全连接层 最后输出每层channel的权值
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            HardSwish(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)       # 不管当前的 h,w 为多少, 全部池化为1
        out = out.view(b, -1)    # 打平处理，与全连接层相连
        # 获取注意力机制后的权重
        out = self.SEblock(out)
        # out是每层channel的权重，需要扩维才能与原特征矩阵相乘
        out = out.view(b, c, 1, 1)  # 增维
        return out * x

class SEInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, activate, use_se, se_kernel_size=1):
        super(SEInvertedBottleneck, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.in_channels = in_channels
        self.out_channels = out_channels
        # mid_channels = (in_channels * expansion_factor)

        # 普通1x1卷积升维操作
        self.conv = Conv1x1BNActivation(in_channels, mid_channels,activate)

        # DW卷积 维度不变，但可通过stride改变尺寸 groups=in_channels
        self.depth_conv = ConvBNActivation(mid_channels, mid_channels, kernel_size,stride,activate)

        # 注意力机制的使用判断
        if self.use_se:
            self.SEblock = SqueezeAndExcite(mid_channels, mid_channels, se_kernel_size)

        # PW卷积 降维操作
        self.point_conv = Conv1x1BNActivation(mid_channels, out_channels,activate)

        # shortcut的使用判断
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        # DW卷积
        out = self.depth_conv(self.conv(x))
        # 当 use_se=True 时使用注意力机制
        if self.use_se:
            out = self.SEblock(out)
        # PW卷积
        out = self.point_conv(out)
        # 残差操作
        # 第一种: 只看步长，步长相同shape不一样的输入输出使用1x1卷积使其相加
        # out = (out + self.shortcut(x)) if self.stride == 1 else out
        # 第二种: 同时满足步长与输入输出的channel, 不使用1x1卷积强行升维
        out = (out + x) if self.stride == 1 and self.in_channels == self.out_channels else out

        return out


class MobileNetV3(nn.Module):
    def __init__(self, num_classes1=4,num_classes2=3):
        super(MobileNetV3, self).__init__()


        # 224x224x3 conv2d 3 -> 16 SE=False HS s=2
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            HardSwish(inplace=True),
        )
        # torch.Size([1, 16, 112, 112])



        self.small_bottleneck = nn.Sequential(
            # torch.Size([1, 16, 112, 112]) 16 -> 16 -> 16 SE=False RE s=2
            SEInvertedBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=2,activate='relu', use_se=True, se_kernel_size=56),
            # torch.Size([1, 16, 56, 56])   16 -> 72 -> 24 SE=False RE s=2
            SEInvertedBottleneck(in_channels=16, mid_channels=72, out_channels=24, kernel_size=3, stride=2,activate='relu', use_se=False),
            # torch.Size([1, 24, 28, 28])   24 -> 88 -> 24 SE=False RE s=1
            SEInvertedBottleneck(in_channels=24, mid_channels=88, out_channels=24, kernel_size=3, stride=1,activate='relu', use_se=False),
            # torch.Size([1, 24, 28, 28])   24 -> 96 -> 40 SE=True RE s=2
            SEInvertedBottleneck(in_channels=24, mid_channels=96, out_channels=40, kernel_size=5, stride=2,activate='hswish', use_se=True, se_kernel_size=14),
            # torch.Size([1, 40, 14, 14])   40 -> 240 -> 40 SE=True RE s=1
            SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
            # torch.Size([1, 40, 14, 14])   40 -> 240 -> 40 SE=True RE s=1
            SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
            # torch.Size([1, 40, 14, 14])   40 -> 120 -> 48 SE=True RE s=1
            SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=48, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
            # torch.Size([1, 48, 14, 14])   48 -> 144 -> 48 SE=True RE s=1
            SEInvertedBottleneck(in_channels=48, mid_channels=144, out_channels=48, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
            # torch.Size([1, 48, 14, 14])   48 -> 288 -> 96 SE=True RE s=2
            SEInvertedBottleneck(in_channels=48, mid_channels=288, out_channels=96, kernel_size=5, stride=2,activate='hswish', use_se=True, se_kernel_size=7),
            # torch.Size([1, 96, 7, 7])     96 -> 576 -> 96 SE=True RE s=1
            SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=7),
            # torch.Size([1, 96, 7, 7])     96 -> 576 -> 96 SE=True RE s=1
            SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=7),
        )

        # torch.Size([1, 96, 7, 7])
        # 相比MobileNetV2，尾部结构改变，,变得更加的高效
        self.small_last_stage = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=576, kernel_size=1, stride=1),
            nn.BatchNorm2d(576),
            HardSwish(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Conv2d(in_channels=576, out_channels=1280, kernel_size=1, stride=1),
            HardSwish(inplace=True),
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 120),
            nn.ReLU(),
            Dropout(0.4),
            nn.Linear(120, num_classes1)
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 120),
            nn.ReLU(),
            Dropout(0.4),
            nn.Linear(120, num_classes2)
        )

        self.init_params()

    # 初始化权重
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)      # torch.Size([1, 16, 112, 112])
       

        x = self.small_bottleneck(x)    # torch.Size([1, 96, 7, 7])
        x = self.small_last_stage(x)  # torch.Size([1, 1280, 1, 1])
        x = x.view(x.size(0), -1)   # torch.Size([1, 1280])
        x1 = self.classifier1(x)      # torch.Size([1, 5])
        x2 = self.classifier2(x)
        return x1, x2
