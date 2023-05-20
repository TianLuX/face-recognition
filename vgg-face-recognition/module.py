# coding:utf-8
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as f
import math
from torch.nn import Parameter
from torchvision.models.resnet import resnet50

# VGG16
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        #默认输入大小为224*224*3
        #特征提取
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #全连接层
        self.affine = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 512, out_features=512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    # 对权重初始化
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 对卷积核进行初始化，使用he初始值
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 对全连接层进行初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # 神经网络的前向传播:
    def forward_once(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)  # 对卷积输出结果按照列来进行展平
        return x

    # 为对比训练做准备:
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        #展平之后做全连接
        output = torch.abs(output1 - output2)

        output = self.affine(output)
        output = output.flatten()

        return output


# 对比训练使用损失函数ContrastiveLoss,一个输入（卷积之后做差，再展平全连接）
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        #进行升维度处理
        label = label.unsqueeze(1)
        output = output.unsqueeze(1)

        loss = torch.nn.BCELoss()
        loss_contrastive = loss(output, label)

        return loss_contrastive
