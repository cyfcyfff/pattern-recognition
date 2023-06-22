import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
from torch.nn.modules.dropout import Dropout

# device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') #（华为云用）
device = torch.device('cpu')


class LeNet(nn.Module):
    def __init__(self, class_nums_1=4, class_nums_2=3):
        super(LeNet, self).__init__()

        # 卷积模块，提取输入数据的特征
        self.conv = nn.Sequential(
            # 1.输入通道数为3，输出通道数为6，卷积核大小为5x5，步长为1的卷积层
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            # 2.使用带有负斜率的ReLU激活函数
            nn.LeakyReLU(0.1),
            # 3.使用大小为2x2，步长为2的最大池化层进行下采样。
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 4.使用输入通道数为6，输出通道数为16，卷积核大小为5x5，步长为1的卷积层
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            # 5.使用带有负斜率的ReLU激活函数
            nn.LeakyReLU(0.1),
            # 6.大小为2x2，步长为2的最大池化层进行下采样
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        # 全连接层模块，对特征进行进一步处理和降维（分类）
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),     # 输入特征的大小为16x4x4=256，输出特征的大小为120的全连接层
            nn.LeakyReLU(0.1),              # 带有负斜率的ReLU激活函数
            nn.Linear(120, 84),             # 输入特征的大小为120，输出特征的大小为84的全连接层
            nn.ReLU(),                      # ReLU激活函数
            nn.Linear(84, 9)             # 输入特征的大小为84，输出特征的大小为9的全连接层
        )

        # 分类器1
        self.classfiler1 = nn.Sequential(
            nn.Linear(45 * 45 * 16, 1000),  # 输入特征向量的大小为45x45x16=32400，输出特征向量的大小为1000的全连接层（同上）
            nn.ReLU(),                      # ReLU激活函数
            nn.Linear(1000, 256),           # 输入特征向量的大小为1000，输出特征向量的大小为256的全连接层
            nn.ReLU(),                      # ReLU激活函数
            Dropout(0.4),                   # 训练过程中以0.4的概率随机将输入的某些元素设为0，用于防止过拟合
            nn.Linear(256, class_nums_1)    # 输入特征向量的大小为256，输出特征向量的大小为class_nums_1的全连接层
        )

        # 分类器2
        self.classfiler2 = nn.Sequential(
            nn.Linear(45 * 45 * 16, 1000),  # 输入特征向量的大小为45x45x16=32400，输出特征向量的大小为1000的全连接层（同上）
            nn.ReLU(),                      # ReLU激活函数
            nn.Linear(1000, 256),           # 输入特征向量的大小为1000，输出特征向量的大小为256的全连接层
            nn.ReLU(),                      # ReLU激活函数
            Dropout(0.4),                   # 训练过程中以0.4的概率随机将输入的某些元素设为0，用于防止过拟合
            nn.Linear(256, class_nums_2)    # 输入特征向量的大小为256，输出特征向量的大小为class_nums_2的全连接层
        )

    # 前向传播函数
    def forward(self, img):
        feature = self.conv(img)        # 将输入的图像数据img经过卷积层和池化层，提取出特征向量feature
        x = torch.flatten(feature, 1)   # 将特征向量feature展平成一个一维的特征向量x，用于后续的全连接层处理
        c1 = self.classfiler1(x)        # 将特征向量x输入到self.classfiler1中进行处理，得到第一个分类任务的预测结果c1
        c2 = self.classfiler2(x)        # 将特征向量x输入到self.classfiler2中进行处理，得到第二个分类任务的预测结果c2
        return c1, c2                   # 输出结果


# 计算准确率
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没有指定device就用net的device
        device = list(net.parameters())[0].device

    acc_sum, n = 0.0, 0
    with torch.no_grad():       # 禁用梯度计算和自动求导，减少显存占用和加速计算
        for X, y in data_iter:
            net.eval()  # 评估模式，这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()     # 累加预测正确的样本
            net.train()     # 训练网络
            n += y.shape[0]

    return acc_sum / n
